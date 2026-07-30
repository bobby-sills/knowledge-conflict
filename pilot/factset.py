"""Fact set construction.

Two sources:

`popqa` (default) — PopQA gives (subj, prop, obj) triplets plus `s_pop`, the
subject's monthly Wikipedia pageviews. That popularity annotation is why the
spec picks it: Test 2 needs log-popularity as the "is this just a frequency
detector?" control, and no other entity-centric set ships it.

`tristate` — the GRD repo's TriState-Bench, which already contains
prior-screened correction/resistance/agreement splits **for our exact model**
(`data/TriState/Meta-Llama-3-8B-Instruct/`). It has no popularity annotation and
only one distractor per fact, so it cannot serve Test 1's candidate-set
knowledge score or Test 2's popularity control. It is loaded as a cross-check
corpus, and because running Test 3 on it makes our numbers directly comparable
to the published GRD table. See DECISIONS.md ("fact set").

Inside-Out (arXiv 2503.15299) has no public fact set that we could find, so the
spec's preference for it is moot; recorded in DECISIONS.md.

PopQA has no distractor sets, so we build them: for each fact, sample other
objects observed for the *same relation*, preferring ones whose own popularity
is close to the gold object's. Popularity-matched distractors matter because an
unmatched set lets any frequency-sensitive scorer look knowledgeable — it would
just be ranking the common entity first.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from collections import defaultdict
from typing import Iterable, Sequence

from . import config
from .splits import assign_split


# --------------------------------------------------------------------------- #
# PopQA
# --------------------------------------------------------------------------- #

def _parse_list(raw) -> list[str]:
    """PopQA stores list columns as strings; some rows use single quotes."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if not s or s in {"[]", "None", "nan"}:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            val = parser(s)
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            return [str(val)]
        except Exception:
            continue
    return [s]


def load_popqa(limit: int | None = None, cache_dir=None) -> list[dict]:
    """Raw PopQA rows, normalised to the fields we use."""
    from datasets import load_dataset
    ds = load_dataset(config.POPQA_HF_ID, split=config.POPQA_SPLIT,
                      cache_dir=str(cache_dir) if cache_dir else None)
    rows = []
    for r in ds:
        rows.append({
            "popqa_id": int(r["id"]),
            "subject": str(r["subj"]),
            "relation": str(r["prop"]),
            "object": str(r["obj"]),
            "subject_id": int(r["subj_id"]),
            "object_id": int(r["obj_id"]),
            "question": str(r["question"]),
            "gold_aliases": _parse_list(r.get("possible_answers")) or [str(r["obj"])],
            "s_pop": int(r["s_pop"]),
            "o_pop": int(r["o_pop"]),
            "s_wiki_title": str(r.get("s_wiki_title") or r["subj"]),
        })
        if limit is not None and len(rows) >= limit:
            break
    return rows


# --------------------------------------------------------------------------- #
# Distractors
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def build_distractor_pools(rows: Sequence[dict]) -> dict[str, list[dict]]:
    """relation -> unique candidate objects observed for that relation."""
    pools: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        pools[r["relation"]][r["object_id"]] = {
            "object": r["object"], "object_id": r["object_id"], "o_pop": r["o_pop"]}
    return {rel: list(d.values()) for rel, d in pools.items()}


def sample_distractors(row: dict,
                       pool: Sequence[dict],
                       forbidden: set[str],
                       n: int = config.N_DISTRACTORS,
                       tol: float = config.DISTRACTOR_POP_TOLERANCE,
                       rng: random.Random | None = None) -> list[dict]:
    """`n` wrong objects for one fact, popularity-matched where possible.

    Two-tier draw: first from candidates whose log10 popularity is within `tol`
    of the gold object's, then top up from the rest of the relation's pool. The
    top-up keeps the candidate-set size constant across facts, which matters
    because the Inside-Out knowledge score is a mean over candidate pairs and an
    uneven denominator would make per-fact scores incomparable.
    """
    rng = rng or random.Random(config.DISTRACTOR_SEED)
    gold_pop = math.log10(max(row["o_pop"], 1))
    near, far = [], []
    for cand in pool:
        if cand["object_id"] == row["object_id"]:
            continue
        if _norm(cand["object"]) in forbidden:
            continue
        gap = abs(math.log10(max(cand["o_pop"], 1)) - gold_pop)
        (near if gap <= tol else far).append((gap, cand))
    rng.shuffle(near)
    rng.shuffle(far)
    far.sort(key=lambda t: t[0])
    picked = [c for _, c in near[:n]]
    if len(picked) < n:
        picked += [c for _, c in far[: n - len(picked)]]
    return picked


# --------------------------------------------------------------------------- #
# Fact records
# --------------------------------------------------------------------------- #

def build_factset(target: int = config.TARGET_FACTS,
                  n_distractors: int = config.N_DISTRACTORS,
                  seed: int = config.DISTRACTOR_SEED,
                  cache_dir=None) -> list[dict]:
    """The canonical fact list, with candidate sets and split assignments.

    Sampling: PopQA is 14k facts over 16 relations and we want ~2k. We stratify
    by relation (proportional to the relation's share of PopQA) so no single
    relation dominates, and sample entities rather than facts so that a subject
    contributing several facts contributes all or none of them — otherwise the
    by-entity split would be fighting the sampler.
    """
    rng = random.Random(seed)
    rows = load_popqa(cache_dir=cache_dir)
    pools = build_distractor_pools(rows)

    # ---- group by subject so entities move as units ---------------------- #
    by_subject: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject_id"]].append(r)

    # ---- stratified subject sample --------------------------------------- #
    subj_relation: dict[int, str] = {
        sid: rs[0]["relation"] for sid, rs in by_subject.items()}
    by_relation: dict[str, list[int]] = defaultdict(list)
    for sid, rel in subj_relation.items():
        by_relation[rel].append(sid)

    n_total_subjects = len(by_subject)
    chosen: list[int] = []
    for rel, sids in sorted(by_relation.items()):
        rng.shuffle(sids)
        quota = max(1, round(target * len(sids) / n_total_subjects))
        chosen.extend(sids[:quota])
    rng.shuffle(chosen)

    facts: list[dict] = []
    for sid in chosen:
        for r in by_subject[sid]:
            if len(facts) >= target:
                break
            forbidden = {_norm(a) for a in r["gold_aliases"]} | {_norm(r["object"])}
            # Any object that is also correct for this subject+relation.
            for other in by_subject[sid]:
                if other["relation"] == r["relation"]:
                    forbidden |= {_norm(a) for a in other["gold_aliases"]}
            dis = sample_distractors(r, pools[r["relation"]], forbidden,
                                     n=n_distractors, rng=rng)
            if len(dis) < n_distractors:
                continue    # relation pool too small; skip rather than shrink
            subject_key = f"popqa:subj:{r['subject_id']}"
            facts.append({
                "fact_id": f"popqa-{r['popqa_id']}",
                "source": "popqa",
                "subject": r["subject"],
                "subject_key": subject_key,
                "relation": r["relation"],
                "object": r["object"],
                "question": r["question"],
                "gold_aliases": r["gold_aliases"],
                "distractors": [d["object"] for d in dis],
                "distractor_pop": [d["o_pop"] for d in dis],
                "s_pop": r["s_pop"],
                "o_pop": r["o_pop"],
                "log_s_pop": round(math.log10(max(r["s_pop"], 1)), 4),
                "s_wiki_title": r["s_wiki_title"],
                "split": assign_split(subject_key),
            })
        if len(facts) >= target:
            break
    return facts


def candidates(fact: dict) -> list[str]:
    """Candidate set in a fixed order: gold first, then distractors."""
    return [fact["object"]] + list(fact["distractors"])


def correct_mask(fact: dict) -> list[bool]:
    """Which candidates count as correct (index-aligned with `candidates`)."""
    golds = {_norm(a) for a in fact["gold_aliases"]} | {_norm(fact["object"])}
    return [_norm(c) in golds for c in candidates(fact)]


def is_hit(text: str, fact: dict) -> bool:
    """Does a free-form generation contain the gold answer?

    Substring-after-normalisation, matching the vendored repo's `substring_match`
    so that our prior-screening and their EM agree about what counts as the
    answer appearing.
    """
    from .vendor import substring_match
    return any(substring_match(a, text) for a in
               (list(fact["gold_aliases"]) + [fact["object"]]))


def pop_bin(s_pop: int, edges: Iterable[float] = config.POP_BIN_EDGES) -> str:
    edges = list(edges)
    for lo, hi in zip(edges, edges[1:]):
        if lo <= s_pop < hi:
            return f"[{int(lo)},{'inf' if hi == float('inf') else int(hi)})"
    return "unbinned"


# --------------------------------------------------------------------------- #
# TriState-Bench (cross-check corpus)
# --------------------------------------------------------------------------- #

TRISTATE_STATE_BY_FILE = {
    "C_right_P_wrong.jsonl": "correction",
    "C_wrong_P_right.jsonl": "resistance",
    "C_right_P_right.jsonl": "agreement",
}


def load_tristate(vendor_root, model_short: str = "Meta-Llama-3-8B-Instruct") -> list[dict]:
    """TriState-Bench for our model, in our fact-record shape.

    Their format is dual-process: two lines per example sharing `input_index`,
    `assigned_process` 0 = context prompt, 1 = question-only prompt. States come
    from the filename, and are *already screened against this model's prior* via
    their GAPS procedure — which is the same job our stage 01 does, by a
    different method. Comparing the two labelings is a free sanity check on our
    6/8 and 0/8 thresholds.
    """
    from pathlib import Path
    base = Path(vendor_root) / "data" / "TriState" / model_short
    if not base.is_dir():
        raise FileNotFoundError(f"no TriState data at {base}")
    out: list[dict] = []
    for fname, state in TRISTATE_STATE_BY_FILE.items():
        path = base / fname
        if not path.exists():
            continue
        by_idx: dict[int, dict] = defaultdict(dict)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_idx[r["input_index"]][r["assigned_process"]] = r
        for idx, pair in sorted(by_idx.items()):
            ctx_row, pri_row = pair.get(0), pair.get(1)
            if ctx_row is None or pri_row is None:
                continue
            golds = ctx_row["gold_answers"]
            golds = golds if isinstance(golds, list) else [golds]
            wrong = ctx_row.get("wrong_answer")
            out.append({
                "fact_id": f"tristate-{state}-{ctx_row.get('fact_id', idx)}",
                "source": "tristate",
                "subject": None,
                "subject_key": f"tristate:{ctx_row.get('fact_id', idx)}",
                "relation": ctx_row.get("answer_type"),
                "object": golds[0],
                "question": pri_row["context_string"],
                "gold_aliases": golds,
                "distractors": [wrong] if wrong else [],
                "distractor_pop": [],
                "s_pop": None,
                "o_pop": None,
                "log_s_pop": None,
                "split": assign_split(f"tristate:{ctx_row.get('fact_id', idx)}"),
                # TriState ships finished prompts; keep them verbatim so the
                # vendored baselines see exactly what the paper fed them.
                "prompt_ctx_raw": ctx_row["context_string"],
                "prompt_pri_raw": pri_row["context_string"],
                "state": state,
                "doc_variant": "faithful" if state != "resistance" else "corrupted",
                "wrong_answer": wrong,
            })
    return out
