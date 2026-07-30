"""Stage 08 — two-pass wall-clock overhead. GPU.

The spec's open question: is the "2x cost" framing in the contrastive-decoding
literature real? Nearly free to answer while the model is already loaded, so we
answer it rather than repeating the claim.
"""

from __future__ import annotations

import argparse

from .. import config, io_utils


def run(n_pairs: int = 16, max_new_tokens: int = config.GEN_MAX_NEW_TOKENS) -> dict:
    paths = config.PATHS.mkdirs()
    facts = {f["fact_id"]: f for f in io_utils.read_jsonl(paths.factset)}
    cases = io_utils.read_jsonl(paths.states)
    if not cases:
        raise SystemExit("run stages 00 and 01 first")

    with io_utils.stage("08_timing", paths.manifest) as box:
        from ..documents import context_prompt, prior_prompt
        from ..model import load_model
        from ..timing import measure

        bundle = load_model()
        pairs = []
        for case in cases[:n_pairs]:
            fact = facts.get(case["fact_id"])
            if fact is None:
                continue
            pairs.append((context_prompt(fact, case["document"], bundle.tokenizer),
                          prior_prompt(fact, bundle.tokenizer)))

        results = measure(bundle, pairs, max_new_tokens=max_new_tokens)
        io_utils.write_json(paths.test_dir("timing") / "timing.json", results)
        box.update({k: v for k, v in results.items() if not k.startswith("_")})

        print("[08] per-sequence latency, relative to plain greedy:")
        for name in ("single", "single_batched2", "two_pass_batched",
                     "two_pass_serial"):
            r = results[name]
            print(f"       {name:<18} {r['sec_per_sequence'] * 1000:7.1f} ms  "
                  f"({r['overhead_vs_single']:.2f}x)  "
                  f"{r['ms_per_token']:.1f} ms/token")
        batched = results["two_pass_batched"]["overhead_vs_single"]
        serial = results["two_pass_serial"]["overhead_vs_single"]
        print(f"[08] batched two-pass costs {batched:.2f}x, serial {serial:.2f}x. "
              f"The literature's '2x' describes the serial form"
              f"{' and holds' if batched > 1.7 else ', not the batched one'}.")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=16)
    a = ap.parse_args(argv)
    return run(a.n_pairs)


if __name__ == "__main__":
    main()
