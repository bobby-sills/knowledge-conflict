"""Bridge to the published baselines. We do not reimplement CAD/AdaCAD/CoCoA/
COIECD/GRD — the spec says use the authors' repo, and a reimplementation is a
reviewer's first target.

Repo: github.com/keith-Jiang/Gated-Reversal-Decoding (renamed from
`conflict-aware-decoding`, the name the spec cites), pinned at
config.VENDOR_COMMIT.

What we take:
  * `methods/*.py` — single-step logit combiners with a uniform interface
    (`get_next_token_logits(logits_ctx, logits_prior)`), which slots straight
    into our own generation loop.
  * `evaluation/utils.py` — SQuAD-style normalisation, `exact_match`,
    `substring_match`, `f1_score`. Using theirs, not ours, is what makes our EM
    numbers comparable to their table.
  * `data/TriState/Meta-Llama-3-8B-Instruct/` — prior-screened conflict states
    for our exact model.

What we do NOT take: their `inference.py` / shell harness, which is built around
a dual-GPU launcher and their own JSONL layout. We drive the method objects
directly.

Note on the spec's "ARR": no method by that name exists in the repo, and no
paper we can find defines it. The strongest published method there is **GRD**
(Gated Reversal Decoding), which is what the Test 3 kill criterion is compared
against. Flagged in DECISIONS.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config

_REPO_DIRNAME = "Gated-Reversal-Decoding"


def ensure_repo(vendor_dir: Path | None = None, commit: str = config.VENDOR_COMMIT) -> Path:
    """Clone (or reuse) the pinned repo. Idempotent; safe after a dead session."""
    vendor_dir = Path(vendor_dir or config.PATHS.vendor)
    vendor_dir.mkdir(parents=True, exist_ok=True)
    root = vendor_dir / _REPO_DIRNAME

    if not (root / ".git").exists():
        print(f"[vendor] cloning {config.VENDOR_REPO} -> {root}")
        subprocess.run(["git", "clone", "--quiet", config.VENDOR_REPO, str(root)],
                       check=True)

    have = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if have != commit:
        print(f"[vendor] checking out {commit[:10]} (was {have[:10]})")
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin", commit],
                       check=False)
        subprocess.run(["git", "-C", str(root), "checkout", "--quiet", commit],
                       check=True)
    return root


def add_to_path(root: Path) -> Path:
    """Their modules import as `methods.base` / `evaluation.utils`, so the repo
    root itself has to be importable."""
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load(vendor_dir: Path | None = None) -> Path:
    return add_to_path(ensure_repo(vendor_dir))


# --------------------------------------------------------------------------- #
# Metrics — vendored, with a self-contained fallback
# --------------------------------------------------------------------------- #

def _fallback_normalise(s: str, remove_punctuation: bool = True) -> str:
    import re
    import string
    text = s.lower()
    if remove_punctuation:
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _metrics():
    """Their metric functions if the repo is available, ours otherwise.

    The fallback is a byte-for-byte port of their `normalize_answer`, so the two
    paths agree; it exists only so that Tests 0-2 (which need `substring_match`
    for prior screening) can run before the clone, e.g. in unit tests.
    """
    try:
        from evaluation.utils import (exact_match, f1_score,   # type: ignore
                                      normalize_answer, substring_match)
        return exact_match, substring_match, f1_score, normalize_answer, True
    except Exception:
        def exact_match(prediction: str, ground_truth: str) -> bool:
            return _fallback_normalise(prediction) == _fallback_normalise(ground_truth)

        def substring_match(ground_truth: str, prediction: str) -> bool:
            return _fallback_normalise(ground_truth) in _fallback_normalise(prediction)

        def f1_score(prediction: str, ground_truth: str) -> float:
            from collections import Counter
            pred = _fallback_normalise(prediction).split()
            gold = _fallback_normalise(ground_truth).split()
            common = Counter(pred) & Counter(gold)
            same = sum(common.values())
            if same == 0:
                return 0.0
            p, r = same / len(pred), same / len(gold)
            return 2 * p * r / (p + r)

        return exact_match, substring_match, f1_score, _fallback_normalise, False


def exact_match(prediction: str, ground_truth: str) -> bool:
    return _metrics()[0](prediction, ground_truth)


def substring_match(ground_truth: str, prediction: str) -> bool:
    return _metrics()[1](ground_truth, prediction)


def f1_score(prediction: str, ground_truth: str) -> float:
    return _metrics()[2](prediction, ground_truth)


def normalize_answer(s: str) -> str:
    return _metrics()[3](s)


def using_vendored_metrics() -> bool:
    """True when the metrics above came from the repo. Recorded in the manifest —
    a run scored with the fallback is not comparable to their published table."""
    return _metrics()[4]


def em_any(prediction: str, golds) -> bool:
    return any(exact_match(prediction, g) for g in golds)


def substring_any(prediction: str, golds) -> bool:
    return any(substring_match(g, prediction) for g in golds)


# --------------------------------------------------------------------------- #
# Baseline decoding methods
# --------------------------------------------------------------------------- #

def build_methods(names=config.VENDOR_METHODS, **kwargs) -> dict:
    """Instantiate the vendored method objects.

    Returns name -> object. Everything except `grd` exposes
    `get_next_token_logits(logits_ctx, logits_prior)`; `grd` is stateful and
    exposes `select_next_token(...)` plus `reset()`. `powerfamily.generate`
    handles both shapes.
    """
    from methods.adacad import AdaCADDecoding      # type: ignore
    from methods.cad import CADDecoding            # type: ignore
    from methods.cocoa import CoCoADecoding        # type: ignore
    from methods.coiecd import COIECDDecoding      # type: ignore
    from methods.greedy import GreedyDecoding      # type: ignore
    from methods.greedy_no_ctx import GreedyNoCtxDecoding  # type: ignore
    from methods.grd import GRDDecoding            # type: ignore

    factories = {
        "greedy": lambda: GreedyDecoding(),
        "greedy_no_ctx": lambda: GreedyNoCtxDecoding(),
        "cad": lambda: CADDecoding(alpha=kwargs.get("cad_alpha", 0.5)),
        "adacad": lambda: AdaCADDecoding(),
        "cocoa": lambda: CoCoADecoding(),
        "coiecd": lambda: COIECDDecoding(),
        "grd": lambda: GRDDecoding(grd_lambda=kwargs.get("grd_lambda", 0.75)),
    }
    out = {}
    for name in names:
        if name not in factories:
            raise KeyError(f"unknown vendored method {name!r}")
        out[name] = factories[name]()
    return out
