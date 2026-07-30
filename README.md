# Internal Knowledge as Arbiter in Knowledge Conflict — pilot

Go/no-go pilot for the question in [`PILOT_SPEC.md`](PILOT_SPEC.md): when a model
reads a document that contradicts what it learned in pretraining, is its
**internal state** a better basis for deciding whom to trust than its **output
probabilities** are?

Five tests, run in order, each with a kill criterion fixed in advance. If one
fires, the pipeline stops. Results go in [`RESULTS.md`](RESULTS.md); every
judgement call the spec left open is recorded in [`DECISIONS.md`](DECISIONS.md).

## 🟠 Outcome: the logit-lens route is dead; the premise is untested.

Run 2026-07-29 on `Meta-Llama-3-8B-Instruct`, A100-40GB, 600 PopQA facts → 683 conflict
cases. Test 1's kill criterion fired and the pipeline stopped.

Internal and external answers **diverge on 50.2% of facts** — five times the 10% floor.
But where they diverge, the **output distribution is right more often than the internal
state: 24.0% vs 19.8%**. And on 56.2% of divergent facts neither is right, so divergence
tracks the model's *ignorance* rather than a correct answer the output layer suppressed.

**Important caveat, found by reading the source paper after the fact.** Test 1 used the
training-free **logit lens**, which the spec chose for cheapness. Inside-Out (arXiv
2503.15299) — the source of the 40% internal-vs-external gap this project is premised on —
does **not** use a logit lens. It uses a **trained linear probe that ingests the candidate
answer**. So the pipeline correctly killed the cheap route, but the premise itself was
never tested. [`DECISIONS.md`](DECISIONS.md) §14 has the side-by-side and the deciding
experiment.

Tests 2–4 were not run. No threshold was adjusted after seeing the result, and the
`report` split is still locked. Both instrument checks passed first, so the null is a
measurement and not an artefact — the lens reconstructs the model's logits to half a
bf16 ULP (wrong convention 89.5× worse), and the final-layer lens matches the external
scores to 5.0e-7.

Full numbers in [`RESULTS.md`](RESULTS.md); what the pilot did *and did not* rule out is
at the end of that file. The reasoning behind every judgement call, including the one
instrument bug that cost a session, is in [`DECISIONS.md`](DECISIONS.md) §10, §13 and §14.

278 tests pass without a GPU, plus 40 slow tests against a real model.

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
pytest -q                                   # 278 tests, no GPU, no model
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
tests/             278 fast tests + 40 slow ones needing --run-slow
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
  error; see DECISIONS.md §10. Measured on the real model: convention `post_norm`,
  reconstruction error **half a bf16 ULP** (the arithmetic floor), the wrong
  convention **89.5×** worse. The tolerance is derived from the compute dtype, not
  hardcoded — an absolute logit tolerance is meaningless across dtypes, and the first
  run of this check died on one that bf16 could not satisfy.
- **Everything is logged on the first pass** — Tests 2 and 4 are re-analyses of
  saved data. `records.py` and `decoding_eval.py` import no torch so that re-analysis
  does not require the capture's environment.
- **Splits are by entity, assigned once** — deterministic hashing, so a rebuilt or
  grown fact set never moves an existing entity.

## Notes on the spec

Resolved in DECISIONS.md §1:

1. The baselines repo has been **renamed** to `keith-Jiang/Gated-Reversal-Decoding`.
2. **ARR is the right comparator, and the pin is not HEAD.** ARR (Adaptive Regime
   Routing) is what arXiv 2606.10298 publishes. On 2026-07-29 the authors deleted
   `methods/arr.py` and replaced it with `methods/grd.py` — a *different* algorithm
   (stateful, and it never extrapolates), not a rename. So we pin `320d88bc`, the
   last commit containing ARR. `config.ARR_RESISTANCE_EM_TARGET` encodes the paper's
   reported 16–33 resistance EM as a reproduction check, because Test 3's kill
   criterion is measured *relative to* this baseline.
3. **Inside-Out released no fact set**, so PopQA it is — which is also what Test 2's
   popularity control needs.

Related: `pilot/regime.py` recovers the τ each baseline is *actually* operating at,
because names and self-reports both lie about it. The vendored `CoCoA.get_tau()`
reports 0.5 while the method runs at α+γ = 1.5 — the interpolation/extrapolation
distinction the paper is about. Stage 06 prints the measured τ for every baseline.
See DECISIONS.md §11.

One bonus: the vendored repo ships **TriState-Bench prior-screened for our exact
model** (~400 cases per conflict state). Not usable as the primary fact set — no
popularity annotation, one distractor per fact — but wired in as a cross-check on
our prior labels and for direct comparability with their published table.
