"""Three-way split, assigned by entity and persisted in the first stage.

Splitting by fact would put the same subject on both sides of the layer-selection
boundary: "Who directed Inception?" and "Who wrote Inception?" share everything
the model knows about Inception. The spec is explicit — split by entity — and
retrofitting it later invalidates results, so it happens in stage 00 and the
assignment is written into factset.jsonl itself.

The hash is deterministic (blake2b of salt + entity key), so a rebuilt fact set
puts the same entity in the same split, and adding facts never moves existing
ones.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from . import config


def entity_bucket(entity_key: str, salt: str = config.SPLIT_SALT) -> float:
    """Stable uniform-ish value in [0, 1) for an entity."""
    h = hashlib.blake2b(f"{salt}::{entity_key}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") / float(1 << 64)


def assign_split(entity_key: str,
                 fractions: dict[str, float] | None = None,
                 salt: str = config.SPLIT_SALT) -> str:
    fractions = fractions or config.SPLIT_FRACTIONS
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    u = entity_bucket(entity_key, salt) * total
    acc = 0.0
    for name in ("train", "layer", "report"):   # fixed order: reproducibility
        acc += fractions[name]
        if u < acc:
            return name
    return "report"


def check_disjoint(rows: Iterable[dict],
                   entity_field: str = "subject_key",
                   split_field: str = "split") -> dict:
    """Verify no entity appears in two splits. Raises if one does."""
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for r in rows:
        ent, sp = r[entity_field], r[split_field]
        counts[sp] = counts.get(sp, 0) + 1
        if ent in seen and seen[ent] != sp:
            raise AssertionError(
                f"entity {ent!r} leaks across splits: {seen[ent]} and {sp}")
        seen[ent] = sp
    return {"n_entities": len(seen), "facts_per_split": counts}


def is_dev(split: str) -> bool:
    return split in config.DEV_SPLITS
