# Internal Knowledge as Arbiter in Knowledge Conflict — pilot

Go/no-go pilot for the question in [`PILOT_SPEC.md`](PILOT_SPEC.md): when a model
reads a document that contradicts what it learned in pretraining, is its
**internal state** a better basis for deciding whom to trust than its **output
probabilities** are?

Five tests, run in order, each with a kill criterion fixed in advance. If one
fires, the pipeline stops. Results go in [`RESULTS.md`](RESULTS.md); every
judgement call the spec left open is recorded in [`DECISIONS.md`](DECISIONS.md).

**Status: built, not yet run.** No model has been executed. 225 tests pass locally
without a GPU; the model-touching stages need a Colab session (see below).

---

## Quick start (Colab)

```python
# Cell 1 — mount Drive and get the code there, so a dead session costs nothing
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone https://github.com/bobby-sills/knowledge-conflict || (cd knowledge-conflict && git pull)
%cd knowledge-conflict
!pip install -q -r requirements.txt
```

```python
# Cell 2 — HF token from Colab Secrets. Llama-3 is a gated repo.
# Add a secret named HF_TOKEN in the sidebar. Never paste a token into a cell.
from pilot.model import hf_token; assert hf_token(), "add HF_TOKEN to Colab Secrets"
```

```python
# Cell 3 — run the pipeline. Safe to re-run after any crash: every stage
# skips what is already on disk.
!python -m pilot.cli all
```

Or stage by stage — see [`notebooks/pilot.ipynb`](notebooks/pilot.ipynb), which is
thin cells calling into the package, with the plots inline.

**Hardware.** Llama-3-8B in bf16 is ~16 GB of weights, so a free-tier T4 (16 GB)
will not hold it with room for activations. Use an **A100 or L4**. Set
`config.SAVE_RESIDUALS = False` (or pass `--no-residuals`) if disk is tight — Tests
1-4 do not need the residual memmap.

## Check the plumbing before spending a GPU session

```bash
pytest -q                                   # 225 tests, no GPU, no model
pytest tests/test_torch_paths.py --run-slow -q   # loads a tiny real model
python -m pilot.cli smoke --model gpt2 --facts 24  # whole pipeline, tiny model
```

The smoke run executes every stage end to end on gpt2 in a scratch directory. Its
numbers are meaningless and it says so; what it validates is that the pipeline runs.

## Where things are

```
pilot/
  config.py        every tunable and every kill threshold, fixed in advance
  io_utils.py      fsynced JSONL, resume, residual memmap, manifest, report lock
  splits.py        by-entity train / layer / report assignment
  factset.py       PopQA + popularity-matched distractors; TriState-Bench loader
  documents.py     faithful / corrupted templates, prompt assembly
  model.py         loading, batched forwards, context-sensitive answer tokenisation
  lens.py          the logit lens — and calibrate(), which verifies it
  prior.py         Test 0 closed-book screening and state labelling
  capture.py       the one expensive pass; logs everything Tests 1-4 need
  records.py       reading capture output back (no torch)
  scoring.py       Test 1 knowledge scores, divergence, internal predictors
  signals.py       Test 2 predictors: 3 internal, 7 external
  stats.py         AUC, paired bootstrap CIs, error-correlation matrix
  analysis.py      assembling the predictor table (no torch)
  reachability.py  Test 3a pairwise reversal thresholds
  powerfamily.py   Test 3b generation, oracle and signal routers
  decoding_eval.py EM, oracle tuning, Test 3 gate (no torch)
  permutation.py   Test 4 control
  vendor.py        the published baselines, pinned
  figures.py       every plot the spec's Report sections name
  timing.py        two-pass overhead measurement
  stages/          one CLI entry point per stage, idempotent and resumable
tests/             225 tests; the torch ones skip without --run-slow
```

## The pipeline

| Stage | Command | Needs GPU | Writes |
|---|---|---|---|
| 00 | `cli factset` | no | `factset.jsonl` |
| 01 | `cli prior` | yes | `prior_samples.jsonl`, `states.jsonl`, Test 0 report |
| 02 | `cli capture` | yes | `capture_prior.jsonl`, `capture_ctx.jsonl`, `resid.npy`, `lens_check.json` |
| 03 | `cli test1` | no | `test1/test1.json` + 2 figures |
| 04 | `cli test2` | no | `test2/test2.json` + 2 figures |
| 05 | `cli test3a` | no | `test3a/test3a.json` + 1 figure |
| 06 | `cli test3b` | yes | `test3b/test3b.json` + 2 figures |
| 07 | `cli test4` | yes* | `test4/test4.json` + 1 figure |
| 08 | `cli timing` | yes | `timing/timing.json` |

\* `cli test4 --skip-decoding` runs the free signal-level control only.

`cli status` shows what exists on disk. Stage 02 is the only expensive one;
everything after it that says "no" is a re-analysis of its output, by design.

## Guardrails, as implemented

The spec's guardrails are enforced in code where prose would not enforce itself:

- **Kill criteria are fixed in advance** — `config.KillCriteria`, written before any
  result existed. `cli all` refuses to run the next stage past a fired criterion.
- **The report split is locked** — `io_utils.assert_report_unlocked` raises unless
  you write the lock file deliberately via `cli unlock-report --reason '...'`.
  Checked before anything else in every analysis stage, so an illegal request cannot
  be masked by a missing-artefact error.
- **Instruments are verified before they are trusted** — `lens.calibrate()` raises if
  the logit lens cannot reproduce the model's own final-layer logits. Stage 02 runs
  it before capturing a single fact. In HF Llama the last hidden state is *already*
  normed, and double-normalising produces plausible wrong numbers rather than an
  error; see DECISIONS.md §10.
- **Everything is logged on the first pass** — Tests 2 and 4 are re-analyses of
  saved data. `records.py` and `decoding_eval.py` import no torch so that re-analysis
  does not require the capture's environment.
- **Splits are by entity, assigned once** — deterministic hashing, so a rebuilt or
  grown fact set never moves an existing entity.

## Notes on the spec

Three things in the spec turned out not to match reality; all are resolved in
DECISIONS.md §1, and one needs a human decision:

1. The baselines repo has been **renamed** to `keith-Jiang/Gated-Reversal-Decoding`.
   Pinned at `581a4d59`.
2. **"ARR" does not exist** in that repo and we could not find a paper defining it.
   We took the spec to mean **GRD** (Gated Reversal Decoding), the repo's own method
   and the strongest comparator there. **Confirm this is what was meant.**
3. **Inside-Out released no fact set**, so PopQA it is — which is also what Test 2's
   popularity control needs.

One bonus: the vendored repo ships **TriState-Bench prior-screened for our exact
model** (~400 cases per conflict state). Not usable as the primary fact set — no
popularity annotation, one distractor per fact — but wired in as a cross-check on
our prior labels and for direct comparability with their published table.
