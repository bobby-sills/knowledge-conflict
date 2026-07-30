# Pilot Study Spec: Internal Knowledge as Arbiter in Knowledge Conflict

**Audience:** Claude Code, implementing in Google Colab.
**Status:** Go/no-go pilot. This is *not* the full project. The purpose is to decide
whether the full project should be built at all.

Read this whole document before writing code. The "Guardrails" section at the end is
as important as the tests.

---

## 1. The research question

When a language model reads a retrieved document that disagrees with what it learned in
pretraining, it has to decide which to follow. Existing methods make that decision from
the model's **output probability distributions**. We want to know whether making it from
the model's **internal state** works better.

The hypothesis in one sentence:

> An LLM's intermediate representations contain information about whether it holds a
> competing belief that its output probabilities do not, and that information is a better
> basis for arbitrating knowledge conflict than output-distribution signals are.

If that is true, the full project is worth building. If it is false, we want to know in
two weeks rather than four months.

## 2. Background you need

**Contrastive decoding for knowledge conflict.** Run the model twice per token: once with
just the question (giving `p_pri`, the parametric/prior distribution), once with the
document in the prompt (giving `p_ctx`). Then decode from a combination. All existing
methods are instances of a single power family:

```
q(y) ∝ p_pri(y)^(1-τ) · p_ctx(y)^τ
```

- `τ = 0` → pure prior. `τ = 1` → pure context.
- `τ ∈ (0,1)` is **interpolation**: bounded, sits between the two distributions.
- `τ > 1` is **extrapolation**: pushes past the context and actively suppresses
  prior-favoured tokens.
- CAD, AdaCAD, COIECD and CoCoA all sit at fixed or adaptively-chosen `τ > 1`. They differ
  only in how they pick `τ`, and they all pick it from output-distribution signals
  (Jensen-Shannon divergence, Rényi divergence, entropy gap, contextual peakedness).

**Reference:** "From Context-Aware to Conflict-Aware" (arXiv 2606.10298). It has public
code at `github.com/keith-Jiang/conflict-aware-decoding` implementing all four baselines
in this unified family. **Use their repo. Do not reimplement the baselines.**

**Three conflict states.** Whether following the context is correct depends on a state
that the decoder cannot see:

| State | Prior | Context | Right action |
|---|---|---|---|
| Correction | wrong | right | follow context (τ high) |
| Resistance | right | wrong | hold the prior (τ low) |
| Agreement | right | right | do nothing (τ ≈ 1) |

Existing methods assume every case is correction. They score below 6 EM on resistance.
That gap is the opportunity.

**Hidden knowledge.** "Inside-Out: Hidden Factual Knowledge in LLMs" (arXiv 2503.15299,
COLM 2025) defines knowledge for a question as the fraction of correct/incorrect answer
pairs in which the correct answer is ranked higher. Score the candidates using output
token probabilities and you get *external* knowledge; score them using intermediate
computations and you get *internal* knowledge. Their finding is that internal exceeds
external — models often know more than their outputs express. Our project is a bet on
that gap being large enough and useful enough to arbitrate conflict.

**Why this could fail.** Recent work (arXiv 2510.09033) argues internal states track
whether a fact was *recalled*, not whether it is *true*. We are predicting knowledge, not
truth, so this may be consistent with our use — but it is a reason the signal might carry
less decision-relevant information than hoped.

## 3. What we are testing, and the order

Five tests. **Run them in order. Stop and report if a kill criterion fires.** Each has a
threshold written down in advance; do not adjust a threshold after seeing a result.

Everything runs on **one model** (`meta-llama/Meta-Llama-3-8B-Instruct`) and a few hundred
to ~2,000 facts. A second model is for the paper, not for this decision.

---

### Test 0 — Does the resistance state exist in useful volume?
*Budget: half a day.*

Build the fact set and label each fact's conflict state for this model.

1. Load an entity-centric fact set of `(subject, relation, object)` triplets with candidate
   answer sets. Start with **PopQA** (has entity popularity annotations, which we need in
   Test 2). Check whether Inside-Out released their fact set — if so, prefer it, since it
   makes our numbers comparable to theirs.
2. **Prior calibration.** For each fact, ask the model closed-book (no document) `N=8`
   times at temperature 0.7. Record whether the gold object appears. Label the prior
   `correct` if it appears in ≥ 6/8 samples, `wrong` if 0/8, `ambiguous` otherwise.
   Discard ambiguous for now but keep the count.
3. **Construct documents** with a fixed template (a short Wikipedia-style passage stating
   the `(subject, relation, object)` fact). Two variants per fact: `faithful` (states the
   gold object) and `corrupted` (states a plausible wrong object drawn from the distractor
   set).
4. Assign states: `correction` = prior wrong + faithful doc. `resistance` = prior correct +
   corrupted doc. `agreement` = prior correct + faithful doc.

**Report:** counts per state; the ambiguous fraction; the distribution of entity popularity
within each state.

**Kill criterion:** none — this test is diagnostic. But if the resistance state can only be
constructed synthetically and is rare under realistic retrieval, flag it prominently.
It changes how the eventual paper must be positioned.

**Note the limitation in your report:** template documents are a weaker evaluation than real
retrieval. Fine for a pilot, must be said out loud.

---

### Test 1 — Do internal and external answers diverge?
*Budget: 2-3 days.*

For each fact, on the **context-free** forward pass (question only, no document), score
every candidate answer two ways:

- **External score:** the model's output token probabilities for that candidate.
- **Internal score:** from intermediate computations. For the pilot, use the **logit lens**:
  take the residual stream at the final question token at each layer `ℓ`, apply the model's
  final layer norm and unembedding matrix, and read off the candidate's probability. This
  requires no training and no labels, which is why we start here.

For each fact and each scoring method compute the Inside-Out knowledge score: the fraction
of (correct, incorrect) candidate pairs where the correct one ranks higher. This is
equivalent to an AUC over the candidate set.

**Report:**
- Internal knowledge score vs external knowledge score, per layer. Which layer maximises
  internal? (Expect mid-to-late.)
- How often internal top-1 ≠ external top-1.
- **When they differ, which is correct more often?**
- Scatter plot: internal score vs external score, coloured by conflict state. This is a
  candidate Figure 1.

**Kill criterion:** divergence below ~10% of facts, **or** internal not beating external on
the subset where they diverge. Either result means the internals carry nothing the output
distribution lacks, and the whole project premise fails.

---

### Test 2 — Does the internal signal predict the *decision*?
*Budget: 1-2 days. **This is the real gate.***

Restrict to cases where the document disagrees with the model — i.e. `correction` and
`resistance`. Frame a binary classification: **should we resist, or should we correct?**
(Ground truth is known from Test 0's state labels.)

Compute AUC on that binary for each of these predictors:

*Internal (ours):*
- internal knowledge score at the best layer from Test 1
- margin between the internal top-1 and top-2 candidates
- layer-trajectory stability: at how many layers is the eventual answer already top-1

*External baselines (must all be included):*
- `H(p_pri)` — output entropy of the prior
- `max p_pri` — top output probability
- CoCoA's entropy gap `H(p_pri) − H(p_ctx)`
- JSD(`p_pri` ‖ `p_ctx`) — AdaCAD's signal
- Rényi divergence — CoCoA's signal
- **log entity popularity** — the "is this just a frequency detector?" control
- self-consistency: agreement rate across the 8 closed-book samples from Test 0

**Report:** AUC table with bootstrap confidence intervals. Plus the **error correlation
matrix** between predictors — this matters even if we lose (see below).

**Kill criterion:** the best internal signal does not beat the best external signal by a
margin you would defend in a paper table (call it +0.05 AUC with non-overlapping CIs).

**If it fails narrowly:** check the error correlation. If the internal signal's errors are
uncorrelated with the external signals' errors, it is *complementary* rather than superior,
and the project becomes an ensemble method. Weaker, still viable. Report this explicitly
rather than reporting a bare failure.

---

### Test 3 — Is there anything to steer toward?
*Budget: 1-2 days.*

Two parts.

**3a. Analytic reachability.** For each conflict case, take the gold token `a` and its
strongest competitor `b`, and compute the pairwise reversal threshold:

```
τ*(a,b) = − ℓ_pri(a,b) / (ℓ_ctx(a,b) − ℓ_pri(a,b))
```

where `ℓ_pri(a,b) = log[p_pri(a)/p_pri(b)]` and likewise for ctx. This is the value of τ at
which the model's preference between `a` and `b` flips. Theory predicts `τ* ∈ (0,1)` for
resistance cases, meaning interpolation is sufficient and nothing exotic is needed. **Verify
that empirically.** Histogram `τ*` by conflict state.

**3b. Oracle ceiling.** Implement the power family and sweep `τ ∈ [0, 2.5]`. Then run
**oracle three-way routing**: pick `τ` per question using the ground-truth state label
(e.g. `τ=1.5` for correction, `τ=0.3` for resistance, `τ=1.0` for agreement; tune these
three constants on a dev split). Compare against CAD, AdaCAD, CoCoA and ARR from the public
repo.

**Report:** trade-off curves (correction accuracy vs resistance accuracy vs agreement
accuracy) for every method, plus the oracle point. Per-state EM.

**Kill criterion:** oracle routing barely beats ARR. No signal quality can exceed the
oracle, so a low ceiling means there is nothing to chase.

**Important:** the oracle must be *three-way routing*, not a one-sided gate that only ever
increases correction strength. A one-sided oracle measures the ceiling of the wrong method.

---

### Test 4 — Permutation control
*Budget: half a day. Cheapest test here and the one most likely to catch a false positive.*

Take whichever signal survived Test 2. Shuffle its values across questions, holding the
marginal distribution fixed. Re-run the τ sweep and the trade-off curves.

**Kill criterion:** the shuffled curve matches the real one. That would mean the signal is
functioning as a global correction-strength knob and the per-question adaptivity — which is
the entire claim — is doing nothing.

**Run this every time a promising result appears, not just once.**

---

## 4. Implementation notes for Colab

**Environment.** `transformers`, `torch`, `datasets`, `accelerate`, `pandas`, `pyarrow`,
`scikit-learn`, `matplotlib`. Load Llama-3-8B-Instruct in bf16. It is a gated repo — the
HF token goes in Colab Secrets, never in a cell.

**Colab sessions die.** Design for that from the first cell:
- Mount Google Drive. All artefacts go to a single project directory there.
- Write results **incrementally to JSONL**, one line per fact, flushed every batch.
- Every stage starts by reading what already exists and skipping completed work. Make every
  script idempotent and resumable. Do not hold results only in notebook state.

**Log everything the first time you touch the model.** Tests 2 and 4 must be *re-analyses of
saved data*, not reruns of the model. For each fact, persist: all candidate scores (internal
per layer, external), all six output-distribution signals, the residual stream at the final
question token for every layer, the closed-book sample outcomes, popularity, state label,
and split assignment. Residual streams at 32 layers × 4096 dims × 2k facts in fp16 is about
0.5 GB — store as a single `.npy` memmap plus a parquet index, not as thousands of files.

**Getting hidden states.** `model(..., output_hidden_states=True)` returns all layers. For
the logit lens apply `model.model.norm` then `model.lm_head` to the layer-`ℓ` hidden state.
Verify your lens implementation by checking that the final layer reproduces the model's
actual logits to within floating-point error. **Do this check before trusting any Test 1
number.**

**Batch the two passes.** The context-free and with-document sequences can go through as a
single batch. Also **measure and report the actual wall-clock overhead** of the two-pass
scheme versus single-pass, batched and unbatched. We have an open question about whether the
"2× cost" framing in the literature is real; this is nearly free to answer while you are here.

**Splits.** Three-way split of the fact set: train / layer-selection / report. Split **by
entity, not by fact** — the same subject must never appear on both sides. Assign splits in
the very first stage and persist the assignment. Retrofitting this later is painful and
invalidates results.

**Structure.** Prefer a small package of importable Python modules with thin notebook cells
that call them, over one long notebook. Notebooks are for plots and narration; logic lives
in `.py` files under version control in the Drive folder.

## 5. Deliverables

1. `results/` — the JSONL/parquet/npy artefacts described above.
2. `figures/` — for each test, the plots named in its Report section.
3. `RESULTS.md` — for each test: the numbers, whether the kill criterion fired, and the
   plots inline. Write this as you go, not at the end.
4. `DECISIONS.md` — a running log of every judgement call you made that the spec did not
   determine (which fact set, distractor construction, popularity binning, the three oracle
   τ constants, and so on), with the reasoning. This is where reviewer questions get
   answered six months from now.

## 6. Guardrails

- **Kill criteria are fixed in advance.** If one fires, stop, write it up, and report. Do
  not adjust a threshold, do not search for a variant that passes, do not proceed to the
  next test. A pilot that kills the project in two weeks has succeeded.
- **Do not touch the report split** until Tests 0-4 are complete. Layer selection uses the
  layer-selection split. Nothing tunes on report.
- **Do not tune anything after seeing test-set results.** If you find yourself wanting to,
  that is a finding to write down, not an action to take.
- **Verify instruments before trusting them.** The logit lens check above is the specific
  example, but the principle is general: any measurement pipeline gets a sanity check
  against a known quantity before its outputs are believed.
- **Negative results are the point.** Test 1 or Test 2 failing is genuinely valuable
  information about the reach of the hidden-knowledge phenomenon. Report it plainly and do
  not editorialise it into a maybe.
- **Flag before assuming.** Raise these rather than deciding silently: whether Inside-Out's
  released fact set is usable; whether ARR's repo runs on our setup; whether the synthetic
  document templates are acceptable or need replacing with ClashEval / ConflictQA /
  ConFiQA instances; whether the closed-book sampling thresholds (6/8, 0/8) are right for
  this model.

## 7. What success looks like

At the end of two weeks we should be able to answer, with numbers:

1. Do internal and external knowledge measurements diverge on this model? By how much?
2. When they diverge, is the internal one right?
3. Does an internal signal predict the resist-or-correct decision better than the six
   output-distribution signals the field currently uses?
4. What is the ceiling if the signal were perfect?
5. Is any apparent gain actually per-question adaptivity, or just a global rescale?

If (1)-(3) come back positive and (4) shows headroom and (5) survives, build the full
project. Otherwise we have learned something real and saved four months.
