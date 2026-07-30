"""Stage 00 — build the fact set and assign splits. CPU only.

Splits are assigned here and written into every fact record. Nothing downstream
re-derives them: one source of truth, fixed before the model is ever loaded.
"""

from __future__ import annotations

import argparse

from .. import config, factset, io_utils, splits


def run(target: int = config.TARGET_FACTS,
        source: str = config.FACTSET_NAME,
        force: bool = False) -> dict:
    paths = config.PATHS.mkdirs()

    if paths.factset.exists() and not force:
        existing = io_utils.read_jsonl(paths.factset)
        if existing:
            print(f"[00] {len(existing)} facts already built; use --force to rebuild")
            return {"n_facts": len(existing), "reused": True,
                    **splits.check_disjoint(existing)}

    with io_utils.stage("00_factset", paths.manifest) as box:
        if source == "popqa":
            facts = factset.build_factset(target=target, cache_dir=paths.cache)
        elif source == "tristate":
            from ..vendor import ensure_repo
            facts = factset.load_tristate(ensure_repo(paths.vendor))
        else:
            raise ValueError(f"unknown fact source {source!r}")

        check = splits.check_disjoint(facts)
        if paths.factset.exists():
            paths.factset.unlink()
        io_utils.append_jsonl(paths.factset, facts)

        by_relation: dict[str, int] = {}
        for f in facts:
            key = str(f.get("relation"))
            by_relation[key] = by_relation.get(key, 0) + 1

        box.update({"n_facts": len(facts), "source": source,
                    "facts_per_relation": by_relation, **check})
        print(f"[00] {len(facts)} facts, {check['n_entities']} entities, "
              f"splits {check['facts_per_split']}")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=config.TARGET_FACTS)
    ap.add_argument("--source", default=config.FACTSET_NAME,
                    choices=("popqa", "tristate"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return run(target=a.target, source=a.source, force=a.force)


if __name__ == "__main__":
    main()
