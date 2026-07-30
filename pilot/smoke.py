"""End-to-end smoke test on a tiny open model. No gated repo, no big GPU.

Purpose: prove the *pipeline* runs — the lens self-check, the resumable IO, the
two-pass generation loop, the analysis stages — before spending an A100 session
on Llama-3-8B. The numbers it produces are meaningless (a 124M-parameter model
knows almost no PopQA facts), and it says so in its own output. What it validates
is plumbing, which is where the time actually goes.

    python -m pilot.cli smoke --model gpt2 --facts 24

Points KC_PROJECT_DIR at a scratch subdirectory so it cannot overwrite a real run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2",
                    help="any small causal LM; gpt2 needs no auth")
    ap.add_argument("--facts", type=int, default=24)
    ap.add_argument("--dir", default=None,
                    help="scratch project dir (default: <repo>/run-smoke)")
    a = ap.parse_args(argv)

    root = Path(a.dir) if a.dir else Path(__file__).resolve().parent.parent / "run-smoke"
    os.environ["KC_PROJECT_DIR"] = str(root)

    from . import config
    config.PATHS = config.Paths(root)
    config.MODEL_ID = a.model
    config.PROMPT_STYLE = "raw"          # gpt2 has no chat template
    config.SYSTEM_PROMPT = ""
    config.TARGET_FACTS = a.facts
    config.PRIOR_N_SAMPLES = 4
    config.PRIOR_MAX_NEW_TOKENS = 8
    config.GEN_MAX_NEW_TOKENS = 8
    config.CAPTURE_BATCH_SIZE = 4
    config.N_DISTRACTORS = 4
    # gpt2 gets almost nothing right closed-book, so the 6/8 rule would label
    # every prior "wrong" and produce zero resistance cases. Loosen it here only —
    # the real run must not touch these.
    config.PRIOR_CORRECT_MIN = 1
    config.PRIOR_WRONG_MAX = 0

    print(f"[smoke] model={a.model} facts={a.facts} dir={root}")
    print("[smoke] NUMBERS FROM THIS RUN ARE MEANINGLESS. It tests plumbing only.\n")

    from .stages import (stage00_factset, stage01_prior, stage02_capture,
                         stage03_test1, stage04_test2, stage05_test3a,
                         stage06_test3b, stage07_test4)

    stage00_factset.run(target=a.facts, force=True)
    stage01_prior.run()
    stage02_capture.run(batch_size=4, residuals=True)

    # Everything is in one split at this scale, so the analysis stages are pointed
    # at whichever splits actually received facts. Guardrails about split hygiene
    # are for the real run; here the goal is to execute every code path.
    from . import io_utils
    facts = io_utils.read_jsonl(config.PATHS.factset)
    present = sorted({f["split"] for f in facts})
    dev = [s for s in present if s != "report"] or present
    print(f"[smoke] splits present: {present}; using {dev}")

    stage03_test1.run(splits_for_report=tuple(dev), splits_for_layer=tuple(dev))
    try:
        stage04_test2.run(fit_split=tuple(dev), report_split=tuple(dev))
    except SystemExit as exc:
        print(f"[smoke] test2 skipped: {exc}")
    stage05_test3a.run(splits=tuple(dev))
    stage06_test3b.run(splits=tuple(dev), grid="coarse", limit=6,
                       methods=("cad", "adacad", "arr"))
    try:
        stage07_test4.run(n_shuffles=2, splits=tuple(dev), limit=6)
    except SystemExit as exc:
        print(f"[smoke] test4 skipped: {exc}")

    print("\n[smoke] pipeline completed. Artefacts:")
    from .cli import status
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
