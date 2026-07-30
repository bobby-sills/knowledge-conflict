"""Crash-tolerant IO. Colab sessions die mid-stage; every stage must be able to
start again and skip what is already on disk.

The contract every stage follows:

    done = load_done_keys(path, key="fact_id")
    for batch in batches(todo minus done):
        ...compute...
        append_jsonl(path, rows)          # flushed + fsynced before returning

Nothing is ever held only in notebook state.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #

def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _sanitise(o: Any) -> Any:
    """NaN/Inf are not valid JSON. Persist them as null rather than emitting
    bare `NaN`, which strict readers (pandas.read_json, jq) reject."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _sanitise(float(o))
    return o


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Append rows and force them to disk. Returns the number written."""
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_sanitise(row), default=_json_default,
                                ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


def read_jsonl(path: Path, skip_bad: bool = True) -> list[dict]:
    """Read a JSONL file, tolerating a truncated final line.

    A session that dies mid-write leaves a partial last line. Dropping it is
    correct: the stage will recompute that row.
    """
    if not Path(path).exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            is_last = i == len(lines) - 1
            if skip_bad and is_last:
                print(f"[io] dropping truncated final line of {Path(path).name}")
                continue
            if skip_bad:
                print(f"[io] skipping unparseable line {i} of {Path(path).name}")
                continue
            raise
    return out


def load_done_keys(path: Path, key: str = "fact_id") -> set:
    """Keys already present in a JSONL artefact."""
    return {r[key] for r in read_jsonl(path) if key in r}


def rewrite_jsonl(path: Path, rows: Sequence[dict]) -> None:
    """Atomic full rewrite (used only for dedup/repair, never in the hot path)."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_sanitise(row), default=_json_default,
                                ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def dedup_jsonl(path: Path, key: str = "fact_id") -> int:
    """Keep the last row per key. Returns the number of duplicates removed.

    Duplicates happen when a session dies after computing a batch but before the
    stage's done-set was reloaded. Harmless, but analysis code assumes one row
    per fact.
    """
    rows = read_jsonl(path)
    if not rows:
        return 0
    seen: dict[Any, dict] = {}
    for r in rows:
        if key in r:
            seen[r[key]] = r
    if len(seen) == len(rows):
        return 0
    rewrite_jsonl(path, list(seen.values()))
    return len(rows) - len(seen)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_sanitise(obj), fh, default=_json_default, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Residual-stream store: one growable memmap + a parquet row index
# --------------------------------------------------------------------------- #

class ResidualStore:
    """A single .npy memmap of shape (n_rows, n_layers, hidden) in fp16.

    32 layers x 4096 dims x 2k facts x 2 bytes ~= 0.5 GB, which is one file, not
    thousands. Row order is *not* fact order: the index parquet maps fact_id ->
    row. Rows are only ever appended, so a resumed session keeps prior rows valid.
    """

    def __init__(self, path: Path, index_path: Path,
                 n_layers: int | None = None, hidden: int | None = None):
        self.path = Path(path)
        self.index_path = Path(index_path)
        self.meta_path = Path(str(path) + ".meta.json")
        meta = read_json(self.meta_path)
        if meta is None:
            if n_layers is None or hidden is None:
                raise ValueError("new ResidualStore needs n_layers and hidden")
            meta = {"n_layers": int(n_layers), "hidden": int(hidden),
                    "dtype": "float16", "n_rows": 0}
            write_json(self.meta_path, meta)
        else:
            if n_layers is not None and meta["n_layers"] != int(n_layers):
                raise ValueError(
                    f"residual store layer mismatch: on disk {meta['n_layers']}, "
                    f"requested {n_layers}. Delete {self.path} to rebuild.")
            if hidden is not None and meta["hidden"] != int(hidden):
                raise ValueError(
                    f"residual store hidden-dim mismatch: on disk {meta['hidden']}, "
                    f"requested {hidden}. Delete {self.path} to rebuild.")
        self.meta = meta

    @property
    def row_bytes(self) -> int:
        return self.meta["n_layers"] * self.meta["hidden"] * 2

    @property
    def n_rows(self) -> int:
        if not self.path.exists():
            return 0
        return self.path.stat().st_size // self.row_bytes

    def append(self, block: np.ndarray) -> tuple[int, int]:
        """Append (b, n_layers, hidden). Returns the [start, end) row range."""
        block = np.ascontiguousarray(block.astype(np.float16, copy=False))
        exp = (self.meta["n_layers"], self.meta["hidden"])
        if block.ndim != 3 or block.shape[1:] != exp:
            raise ValueError(f"expected (b, {exp[0]}, {exp[1]}), got {block.shape}")
        start = self.n_rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as fh:
            fh.write(block.tobytes(order="C"))
            fh.flush()
            os.fsync(fh.fileno())
        end = self.n_rows
        if end != start + block.shape[0]:
            raise IOError(f"short write: {start} -> {end}, expected +{block.shape[0]}")
        self.meta["n_rows"] = end
        write_json(self.meta_path, self.meta)
        return start, end

    def open_memmap(self, mode: str = "r") -> np.memmap:
        n = self.n_rows
        return np.memmap(self.path, dtype=np.float16, mode=mode,
                         shape=(n, self.meta["n_layers"], self.meta["hidden"]))

    def read_rows(self, rows: Sequence[int]) -> np.ndarray:
        mm = self.open_memmap()
        return np.asarray(mm[np.asarray(rows, dtype=np.int64)], dtype=np.float32)

    def truncate_to(self, n_rows: int) -> None:
        """Drop trailing rows (repair after a partial write)."""
        with open(self.path, "r+b") as fh:
            fh.truncate(n_rows * self.row_bytes)
        self.meta["n_rows"] = n_rows
        write_json(self.meta_path, self.meta)

    def write_index(self, index_rows: Sequence[dict]) -> None:
        import pandas as pd
        pd.DataFrame(list(index_rows)).to_parquet(self.index_path, index=False)

    def read_index(self):
        import pandas as pd
        if not self.index_path.exists():
            return pd.DataFrame(columns=["fact_id", "row"])
        return pd.read_parquet(self.index_path)


# --------------------------------------------------------------------------- #
# Manifest — provenance for every stage run
# --------------------------------------------------------------------------- #

def _pkg_versions() -> dict:
    out: dict[str, Any] = {"python": sys.version.split()[0],
                           "platform": platform.platform()}
    for name in ("torch", "transformers", "datasets", "numpy", "pandas", "sklearn"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def _git_rev() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def log_manifest(path: Path, stage: str, extra: dict | None = None) -> dict:
    """Append a provenance record. Called by every stage, at the end."""
    from . import config
    rec = {
        "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.snapshot(),
        "env": _pkg_versions(),
        "code_rev": _git_rev(),
    }
    if extra:
        rec["result"] = extra
    append_jsonl(path, [rec])
    return rec


@contextmanager
def stage(name: str, manifest_path: Path) -> Generator[dict, None, None]:
    """Time a stage and always leave a manifest entry, success or failure."""
    print(f"\n=== stage {name} ===")
    t0 = time.time()
    box: dict = {}
    try:
        yield box
    except BaseException as exc:
        box["error"] = f"{type(exc).__name__}: {exc}"
        box["seconds"] = round(time.time() - t0, 1)
        log_manifest(manifest_path, name, box)
        raise
    box["seconds"] = round(time.time() - t0, 1)
    log_manifest(manifest_path, name, box)
    print(f"=== stage {name} done in {box['seconds']}s ===")


# --------------------------------------------------------------------------- #
# Report-split lock (Guardrail 2)
# --------------------------------------------------------------------------- #

class ReportSplitLocked(RuntimeError):
    pass


def assert_report_unlocked(lock_path: Path, who: str) -> None:
    """Refuse to touch the report split unless it has been explicitly unlocked.

    The guardrail is 'do not touch the report split until Tests 0-4 are
    complete'. Prose does not enforce itself, so this does: analysis defaults to
    the dev splits, and reaching the report split requires writing the lock file
    with an explicit reason.
    """
    if not Path(lock_path).exists():
        raise ReportSplitLocked(
            f"{who} asked for the report split, which is locked.\n"
            f"Tests 0-4 run on the dev splits {('train', 'layer')}.\n"
            f"To unlock (once, deliberately, after Tests 0-4 are written up):\n"
            f"    python -m pilot.cli unlock-report --reason '...'"
        )


def unlock_report(lock_path: Path, reason: str) -> dict:
    rec = {"unlocked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reason": reason}
    write_json(Path(lock_path), rec)
    return rec


def batched(seq: Sequence, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])
