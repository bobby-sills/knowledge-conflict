"""Stage 02 — the one expensive pass. GPU.

Order of operations matters here: the logit lens is calibrated and self-checked
*before* a single fact is captured, and the check's result goes in the manifest.
If the lens cannot reproduce the model's own final-layer logits, this stage
raises and captures nothing — a mis-detected lens must stop the pipeline rather
than fill a 0.5 GB file with numbers that look fine.

Resumable at batch granularity. Residual streams append to a single memmap and
the fact_id -> row index is rewritten each batch, so an interrupted run leaves
the store and its index consistent.
"""

from __future__ import annotations

import argparse

from .. import config, io_utils
from ..capture import capture_ctx_batch, capture_prior_batch


def _prior_path():
    return config.PATHS.results / "capture_prior.jsonl"


def _ctx_path():
    return config.PATHS.results / "capture_ctx.jsonl"


def run(limit: int | None = None,
        batch_size: int = config.CAPTURE_BATCH_SIZE,
        residuals: bool = config.SAVE_RESIDUALS,
        skip_full_span: bool = False) -> dict:
    paths = config.PATHS.mkdirs()
    facts = io_utils.read_jsonl(paths.factset)
    cases = io_utils.read_jsonl(paths.states)
    if not facts or not cases:
        raise SystemExit("run stages 00 and 01 first")

    facts_by_id = {f["fact_id"]: f for f in facts}
    needed_fact_ids = {c["fact_id"] for c in cases}
    facts = [f for f in facts if f["fact_id"] in needed_fact_ids]
    if limit:
        facts = facts[:limit]
        keep = {f["fact_id"] for f in facts}
        cases = [c for c in cases if c["fact_id"] in keep]

    with io_utils.stage("02_capture", paths.manifest) as box:
        from ..lens import calibrate
        from ..model import load_model

        bundle = load_model()
        lens_report = calibrate(bundle)          # raises if the lens is wrong
        box["lens_check"] = lens_report
        io_utils.write_json(paths.results / "lens_check.json", lens_report)

        store = None
        if residuals:
            from ..io_utils import ResidualStore
            store = ResidualStore(paths.resid, paths.resid_index,
                                  n_layers=bundle.n_hidden_states,
                                  hidden=bundle.hidden)

        # ---- prior pass, per fact ---------------------------------------- #
        done = io_utils.load_done_keys(_prior_path(), "fact_id")
        todo = [f for f in facts if f["fact_id"] not in done]
        print(f"[02] prior pass: {len(done)} done, {len(todo)} to go")

        index_rows = [{"fact_id": r["fact_id"], "row": r["resid_row"]}
                      for r in io_utils.read_jsonl(_prior_path())
                      if r.get("resid_row") is not None]
        for chunk in io_utils.batched(todo, batch_size):
            rows, resid = capture_prior_batch(
                bundle, chunk, want_residuals=residuals,
                full_span_scores=not skip_full_span)
            if store is not None and resid is not None:
                start, _ = store.append(resid)
                for offset, row in enumerate(rows):
                    row["resid_row"] = start + offset
                    index_rows.append({"fact_id": row["fact_id"],
                                       "row": row["resid_row"]})
                store.write_index(index_rows)
            io_utils.append_jsonl(_prior_path(), rows)
            print(f"[02] prior {len(io_utils.load_done_keys(_prior_path()))}"
                  f"/{len(facts)}", flush=True)

        io_utils.dedup_jsonl(_prior_path(), "fact_id")
        prior_rows = io_utils.read_jsonl(_prior_path())
        prior_by_id = {r["fact_id"]: r for r in prior_rows}

        # ---- context pass, per case -------------------------------------- #
        done_cases = io_utils.load_done_keys(_ctx_path(), "case_id")
        todo_cases = [c for c in cases
                      if c["case_id"] not in done_cases
                      and c["fact_id"] in prior_by_id]
        print(f"[02] ctx pass: {len(done_cases)} done, {len(todo_cases)} to go")

        for chunk in io_utils.batched(todo_cases, batch_size):
            rows = capture_ctx_batch(bundle, chunk, facts_by_id, prior_by_id)
            io_utils.append_jsonl(_ctx_path(), rows)
            print(f"[02] ctx {len(io_utils.load_done_keys(_ctx_path(), 'case_id'))}"
                  f"/{len(cases)}", flush=True)

        io_utils.dedup_jsonl(_ctx_path(), "case_id")

        usable = sum(1 for r in prior_rows if r.get("usable_first_token"))
        box.update({
            "n_facts_captured": len(prior_rows),
            "n_cases_captured": len(io_utils.load_done_keys(_ctx_path(), "case_id")),
            "n_layers": bundle.n_hidden_states,
            "usable_first_token_fraction": usable / max(len(prior_rows), 1),
            "residual_rows": store.n_rows if store else 0,
        })
        print(f"[02] first-token-usable facts: {usable}/{len(prior_rows)} "
              f"({usable / max(len(prior_rows), 1):.1%})")
        if usable / max(len(prior_rows), 1) < 0.8:
            print("[02] FLAG: many facts have gold/distractor first-token "
                  "collisions. First-token scoring is capped on those facts "
                  "through no fault of the model; report the rate with Test 1.")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=config.CAPTURE_BATCH_SIZE)
    ap.add_argument("--no-residuals", action="store_true",
                    help="skip the 0.5 GB residual memmap (Tests 1-4 do not need "
                         "it; trained probes in the full project would)")
    ap.add_argument("--skip-full-span", action="store_true",
                    help="skip the secondary multi-token external score")
    a = ap.parse_args(argv)
    return run(limit=a.limit, batch_size=a.batch_size,
               residuals=not a.no_residuals, skip_full_span=a.skip_full_span)


if __name__ == "__main__":
    main()
