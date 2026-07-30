# DECISIONS

Every judgement call the spec did not determine, with the reasoning. This is where
reviewer questions get answered six months from now.

Entries added while building the pipeline are marked **[build]**. Entries that must
be filled in once the pipeline has run are marked **[open]** — they are decisions
that depend on seeing data, and the spec's guardrails say to flag rather than
decide silently.

---

## 1. Flags raised before assuming — resolved

### 1.1 The baseline repo has been renamed **[build]**

The spec cites `github.com/keith-Jiang/conflict-aware-decoding`. That repo is now
**`github.com/keith-Jiang/Gated-Reversal-Decoding`** — same authors, same paper
(arXiv 2606.10298). The rename happened on 2026-07-29, alongside the method swap
described in §1.2, so it is a symptom of that rather than a separate event.

We pin `config.VENDOR_COMMIT = "320d88bc"` (2026-07-09), **not** HEAD, because HEAD
no longer contains the method the paper publishes. See §1.2 for why.

### 1.2 ARR is the paper's method; the repo replaced it mid-build **[build]**

**Resolved: the spec is right. ARR is correct and is the Test 3 comparator.** An
earlier version of this file claimed no such method existed, on the basis of a repo
read that was accurate but badly timed. Recording the sequence, because the failure
mode generalises.

What happened:

* arXiv 2606.10298 v1 (9 June 2026, still the only version) names **Adaptive Regime
  Routing (ARR)** in its abstract, crediting it with lifting resistance EM "from
  below 6 to 16--33".
* `methods/arr.py` was present in the repo from its initial commit (8 June 2026)
  through 9 July 2026.
* On **29 July 2026 at 14:08 UTC**, commit `13738f76` deleted `arr.py`, added
  `grd.py` (Gated Reversal Decoding), and the repo was renamed from
  `conflict-aware-decoding` to `Gated-Reversal-Decoding`. Its README was rewritten
  to present GRD as the paper's method, with different headline numbers (2.1 → 20.8
  rather than <6 → 16–33).
* We first read the repo at roughly 16:30 UTC the same day — about two hours after
  the swap — found no mention of ARR anywhere, and wrongly concluded the spec had
  named a method that did not exist.

**GRD is a different algorithm, not a rename.** Diffed directly:

| | ARR (`arr.py`, published) | GRD (`grd.py`, HEAD) |
|---|---|---|
| State | stateless, recomputed per step | stateful; locks the trusted branch at the first conflict step |
| τ | `1 + s` if `p_ctx_max > p_pri_max` else `1 - s`, `s = JSD/log2` clipped to [0,1] | `(1 - λ)·τ*` with λ=0.75, τ* ∈ (0,1), so τ ≤ 0.25 |
| Regime | routes across τ=1 — both extrapolation and interpolation | interpolation only; **never extrapolates** |
| Gate | confidence asymmetry between ctx and prior | `p_prior_top > 0.5` **and** `H(prior) < H(ctx)` |

So GRD cannot be the comparator: the paper's claim is about per-step routing between
regimes, and GRD only ever interpolates.

**What this changes in the code:**

* `config.VENDOR_COMMIT` is pinned to **`320d88bc`** (9 July 2026), the last commit
  containing `arr.py` — *not* HEAD. HEAD does not contain the paper's method. That
  commit also has all four baselines, the metrics, and the TriState data, so nothing
  is lost. `config.VENDOR_HEAD_WITH_GRD` records the HEAD sha without using it.
* `VENDOR_METHODS = ("cad", "adacad", "cocoa", "coiecd", "arr")`. `greedy` and
  `greedy_no_ctx` are dropped because they are exactly `PowerFamily(1.0)` and
  `PowerFamily(0.0)`, already computed by the τ sweep.
* `config.ARR_RESISTANCE_EM_TARGET = (0.16, 0.33)` — the paper's reported range, as
  a **reproduction target**. Stage 06 flags if our ARR run lands outside it. This is
  the check that does not depend on names: Test 3's kill criterion is measured
  *relative to* the best baseline, so a wrong or misconfigured comparator silently
  moves the gate.

**The generalisable lesson.** Verifying against a repo is not verifying against the
paper, and a repo can change under you mid-task. The three functional criteria
(per-step routing across τ=1; gate driven by confidence asymmetry; resistance EM in
16–33) identify the method whatever the file is called, which is why they are now
encoded as a probe and a threshold rather than as a name match. See §11.

If the authors revise the paper to publish GRD, add it back by pointing
`VENDOR_COMMIT` at HEAD and appending `"grd"` to `VENDOR_METHODS`;
`powerfamily.generate` already handles its stateful `select_next_token` interface.

### 1.3 Inside-Out released no fact set **[build]**

The spec prefers Inside-Out's (arXiv 2503.15299) fact set if it exists, to make our
numbers comparable to theirs. We could find no code or data release — not on arXiv,
not on GitHub, not linked from the paper page. So: **PopQA**, as the spec's
fallback. Comparability to Inside-Out is lost; comparability of the *metric* is
kept, since we implement their knowledge-score definition directly.

### 1.4 TriState-Bench already contains prior-screened states for our model **[build]**

Unexpected and useful. The vendored repo ships
`data/TriState/Meta-Llama-3-8B-Instruct/` with three files —
`C_right_P_wrong.jsonl` (correction), `C_wrong_P_right.jsonl` (resistance),
`C_right_P_right.jsonl` (agreement) — about **400 cases each**, screened against
*our exact model* via their Greedy-Anchored Prior Screening (GAPS).

We did **not** adopt it as the primary fact set, because:

* No entity-popularity annotation. Test 2 requires log-popularity as the "is this
  just a frequency detector?" control, and there is nothing to compute it from.
* One distractor per fact (`wrong_answer`). Test 1's knowledge score is a mean over
  (correct, incorrect) candidate pairs; with one incorrect candidate it degenerates
  to a single comparison and is far too noisy per fact.
* The facts are synthetic and generic (`fact_id: person_2`, "Who was the first
  president of the United States?"), so entity popularity is not even well defined.

It **is** wired in (`factset.load_tristate`, `--source tristate`) for two jobs:
1. Cross-checking our 6/8-and-0/8 prior labels against their GAPS labels
   (`prior.agreement_with_tristate`) — the only external validity check on our
   thresholds available without new annotation.
2. Running Test 3 on their data, which makes our EM directly comparable to the
   published table.

### 1.5 Synthetic document templates are a real limitation **[build]**

Stated out loud, as the spec requires. `documents.py` builds one Wikipedia-style
passage per relation, with faithful and corrupted variants differing only in the
stated object. This is weaker than real retrieval in three ways: the passages are
shorter and more assertive than retrieved text; they contain exactly one claim, so
there are no partial or hedged conflicts; and a retriever's failure modes
(irrelevant passages, multiple contradictory passages) are absent entirely.

Fine for a pilot. **Not fine for the paper.** The replacements to evaluate are
ClashEval, ConflictQA and ConFiQA. Flagged, not decided.

Mitigation applied: both variants come from the same template with only the object
swapped, and a test asserts their lengths differ by under 30 characters. Without
that, a less fluent corrupted passage would confound every resistance result with
fluency rather than conflict.

### 1.6 Whether the 6/8 and 0/8 thresholds suit this model — **[open]**

Cannot be decided before running. Stage 01 reports the ambiguous fraction and
prints a prominent flag if it exceeds 50%. If it does, the thresholds are wrong for
this model rather than the model being uncertain — **record the finding here and
raise it, do not retune silently.**

### 1.7 Whether the vendored repo runs on our setup — **[open]**

It pins `torch==2.5.0` / `transformers==4.51.3`; we run on whatever Colab provides.
We only import its pure-logits method classes and its string-normalisation metrics,
neither of which touches a version-sensitive API, so we do not install its pins.
`vendor.using_vendored_metrics()` returns False if the import failed and we fell
back to our own port of `normalize_answer`; stage 06 prints a flag in that case,
because a run scored with the fallback is not comparable to their published table.

---

## 2. Fact set

**PopQA, stratified by relation, sampled by entity.** 14k facts over 16 relations,
target ~2,000. Sampling is stratified by relation proportionally to each relation's
share of PopQA, so no relation dominates. Entities move as units — a subject
contributing several facts contributes all or none — because otherwise the sampler
would be fighting the by-entity split.

**Distractors: 8 per fact, popularity-matched.** PopQA has no distractor sets, so
they are drawn from other objects observed for the *same relation*. The draw is
two-tier: first from candidates whose log10 popularity is within 1.0 of the gold
object's, then topped up from the rest of the relation's pool.

Popularity matching is not cosmetic. An unmatched distractor set lets any
frequency-sensitive scorer look knowledgeable — it would just be ranking the more
common entity first, and both the internal and external scores are
frequency-sensitive. The top-up keeps the candidate-set size fixed at 9, which
matters because the knowledge score's denominator is the number of candidate pairs;
an uneven denominator would make per-fact scores incomparable.

Facts whose relation pool cannot supply 8 distractors are dropped rather than given
a smaller candidate set, for the same reason.

**Excluded from a fact's distractors:** the gold object, all its aliases, and any
object that is correct for the same subject under the same relation.

---

## 3. Scoring granularity — the most consequential decision here

**Primary comparison is first-answer-token.** The logit lens reads exactly one
position: the residual stream at the final question token. So the primary external
score is read at the same position — the output distribution over the *first*
answer token.

Scoring the external side over a whole multi-token span while the internal side
sees one token would compare a sentence scorer to a word scorer and then attribute
the difference to "hidden knowledge". The full-span score (length-normalised
teacher-forced log-prob) is still logged as `external_full_span` and reported as a
secondary, but it is never the headline.

**First tokens are resolved contextually.** `model.first_answer_token_id` tokenises
`prompt` and `prompt + answer` and takes the first token past the prompt's length,
rather than tokenising the answer alone. BPE is context-sensitive: " politician"
tokenises differently depending on what precedes it, and the standalone form is
often not what the model would emit. Where the prompt's tokenisation is not a
prefix of the joint one (a real if rare BPE re-merge), the candidate scores NaN and
the fact is excluded rather than scored against the wrong token.

**First-token collisions are counted, not hidden.** "politician" and "political
leader" share a first token and are indistinguishable to a first-token scorer. Such
facts have a capped achievable knowledge score through no fault of the model.
Stage 02 reports `usable_first_token_fraction` and flags below 80%; stage 03
reports Test 1 both unrestricted (the honest headline) and restricted to
collision-free facts (showing whether collisions drive the result).

---

## 4. Prompt format

**Llama-3 chat template for capture, raw format for the vendored baselines.**

Llama-3-8B-**Instruct** was trained with a chat template, and `PROMPT_STYLE="chat"`
uses it with `add_generation_prompt=True` so the string ends at the assistant
header — the next token is the first answer token, which is where every score is
read. A raw `"Question: ...\nAnswer:"` prompt to an instruct model measures its
behaviour off-distribution.

`PROMPT_STYLE="raw"` reproduces the GRD repo's format and is used when running
their baselines, so our numbers line up with theirs. TriState-Bench cases carry
finished prompts (`prompt_ctx_raw` / `prompt_pri_raw`) which are passed through
verbatim.

The system prompt demands the shortest possible span. Without it an instruct model
answers in sentences and strict EM collapses for reasons unrelated to knowledge
conflict — which is also why substring EM is reported alongside strict EM
everywhere.

---

## 5. Splits

**Three-way by entity: train 0.40 / layer 0.30 / report 0.30.** Assigned in stage
00 from `blake2b(salt + subject_key)` and written into every fact record. Nothing
downstream re-derives them.

Deterministic hashing rather than a shuffled partition, so that rebuilding the fact
set — or growing it — keeps every existing entity in the same split. A shuffle
would silently reassign entities on a rebuild and invalidate every earlier result.
A test asserts this.

**Report split is locked in code, not in prose.** `io_utils.assert_report_unlocked`
raises unless a lock file has been written by `python -m pilot.cli unlock-report
--reason '...'`. The guardrail is "do not touch the report split until Tests 0-4
are complete", and prose does not enforce itself. Every analysis stage checks the
lock *before* anything else, so an illegal split request cannot be masked by a
missing-artefact error.

**Which split each test reports on:**

| Stage | Fits on | Reports on |
|---|---|---|
| 03 / Test 1 | layer (layer choice only) | train |
| 04 / Test 2 | train (divergence directions, thresholds) | layer |
| 05 / Test 3a | — | train + layer |
| 06 / Test 3b | train (oracle τ constants, via the sweep) | train + layer |
| 07 / Test 4 | train (threshold) | layer |

Test 2 reports on the layer split, which Test 1 also used to choose the layer. That
is a mild reuse and a deliberate trade: the alternative is reporting Test 2 on the
report split, which the guardrail forbids until the write-up is done. **The intended
final step, after Tests 0-4 are written up, is to unlock the report split and re-run
the frozen pipeline once on it.** Nothing may be tuned after that.

---

## 6. The knowledge score

**Ties count as 0.5.** A scorer assigning identical scores to a correct and an
incorrect candidate has not ranked the correct one higher. Counting ties as wins
would inflate any coarse or saturated scorer, and the logit lens produces plenty of
near-ties at early layers. This is the standard AUC tie convention and what
Inside-Out's pair-counting definition implies; a test asserts our implementation
equals `sklearn.roc_auc_score` including on tied inputs.

**A NaN anywhere poisons the fact's score.** A partially scored candidate set would
change the pair denominator between facts. NaN, not partial credit.

**Top-1 ties count as wrong.** Gold is always candidate index 0, so `argmax`'s
first-index tie-break would credit every tie to gold and flatter every scorer.

---

## 7. Test 2 predictors

**Ten predictors: three internal, seven external.** The seven external are the six
the spec names plus `ctx_entropy` retained in the capture rows for re-analysis.

**Sign convention: larger always means "resist".** Fixed in
`signals.RESIST_ORIENTATION`. Without one convention, half the AUCs land below 0.5
and "the best signal" stops meaning anything.

**The symmetric divergences get a fitted direction.** JSD and Rényi measure *how
much* two distributions differ, not which to trust; both orientations are equally
defensible a priori. Their direction is fitted on the **train** split and applied
unchanged to the reported split. That is one free parameter granted to a baseline —
the right way to lose an argument with a reviewer rather than win one cheaply.

**Self-consistency is agreement, not accuracy.** The share of the 8 closed-book
samples matching the modal answer, not the share that are correct. Accuracy would
leak the Test 2 label directly into a predictor of that label.

**Bootstrap resamples questions, shared across predictors.** 2,000 draws, 95% CI.
The same resampled indices are used for every predictor, so paired differences are
meaningful; independent resampling would inflate the apparent gap between two
predictors that agree, which is exactly the comparison the gate rests on. Both the
paired-difference CI (the right test for "A beats B") and non-overlapping marginal
CIs (the weaker, more conservative condition the spec named) are reported.

**Error thresholds are fitted on train.** So the correlated error patterns are
out-of-sample errors, not in-sample fits.

**A constant error vector yields NaN correlation, not 0.** A predictor that is
never wrong (or always wrong) has no correlation with anything — 0/0. Reporting 0.0
would read as "uncorrelated errors", which is precisely the finding that keeps the
project alive as an ensemble method, and it must not be manufactured by a division
by zero. Such predictors are listed in `error_correlation.degenerate`.

---

## 8. Test 3

**τ grid: 0.00 to 2.50.** Coarse (11 points, 0.25 spacing) by default, fine (51
points, 0.05) available. A 51-point sweep over thousands of cases is the longest
stage in the pilot and the interesting structure is a crossing, which 11 points
resolve.

**Oracle τ constants are tuned, not fixed.** The spec suggests 1.5 / 0.3 / 1.0 as a
starting point; `decoding_eval.tune_oracle` instead takes the per-state argmax over
the sweep on the train split. Ties break toward τ=1 so a state with a flat curve
does not get a spuriously extreme constant.

**The oracle is assembled from the sweep, not regenerated.** Selecting the row
already generated at each state's routed τ is provably identical to regenerating,
costs nothing, and guarantees the oracle cannot see anything the sweep did not.

**Three-way routing, as the spec insists.** A one-sided gate that only raises
correction strength is bounded above by "always follow the context" and would make
resistance look unreachable regardless of signal quality.

**Generation runs both prompts as a two-row batch sharing one KV cache**, and both
rows are fed the *same* chosen token each step. Feeding each row its own argmax
would let the prior branch wander onto a different sequence, and `p_pri` would stop
being "the prior for this continuation".

---

## 9. Test 4

**Gains are measured against the best fixed τ, not against τ=1.** Against τ=1, a
pure global rescale would look like a win and pass a control designed to catch
exactly that.

**Two shuffle modes.** `permute_within_all` catches a signal acting as a global
knob. `permute_within_state` is stricter: it preserves any between-state difference
in the signal's distribution and asks whether the *within*-state ordering carries
anything. A signal that survives the first only because correction and resistance
cases occupy different signal ranges fails the second, and the gap between them
says which kind of adaptivity is actually present.

**The cheap signal-level control runs on every signal in stage 04**, not only on
the winner, per the guardrail to run it every time a promising result appears.

---

## 10. Engineering decisions

**The logit lens is calibrated, not assumed. [build]** In HuggingFace Llama the
last element of `output_hidden_states` has *already* been through
`model.model.norm`; applying the norm again double-normalises it. RMSNorm is not
idempotent, so the result is wrong but plausible — a valid distribution, peaked on
sensible tokens, quietly not the model's. Every Test 1 number would be off with
nothing to show it.

`lens.calibrate()` reconstructs the final-layer logits both ways, compares against
the logits the model actually returned, records which convention holds, and
**raises** if neither matches within tolerance. Stage 02 calls it before capturing
a single fact and writes the result to `results/lens_check.json`.

**Left padding everywhere, with explicit `position_ids`. [build]** With left padding
the last position of every row is that row's final prompt token, so "the residual
stream at the final question token" needs no per-row index arithmetic. But left
padding also means the first real token sits at index k, not 0 — the default
`0..T-1` positions would rotate every RoPE embedding by k. The model still produces
fluent text, so this is another failure that does not announce itself; it just makes
the two rows of the batch incomparable, which is fatal when the whole method is a
comparison between them.

**Analysis modules import no torch. [build]** `records.py` and `decoding_eval.py`
exist so that Tests 1, 2, 3a and 4 can be re-run on a CPU-only machine. The spec
requires them to be re-analyses of saved data rather than reruns of the model; if
importing the analysis pulled in torch, "re-analysis" would quietly require the same
environment as the capture.

**Residual streams: one memmap plus a parquet index.** 33 states × 4096 dims × 2k
facts in fp16 is 0.54 GB — one file, not thousands, as the spec directs. Rows are
append-only so a resumed session keeps prior rows valid. Not needed by Tests 1-4
(`--no-residuals` skips it); it is there for the trained probes the full project
would use.

**JSONL writes are fsynced; truncated final lines are dropped on read.** A session
killed mid-write leaves a partial last line, and dropping it is correct — the stage
recomputes that row. Duplicate rows (a batch that landed twice before the done-set
reloaded) are deduped keeping the last.

**NaN and Inf serialise as null.** Bare `NaN` is invalid JSON and strict readers
reject it.

**Dependencies are not pinned to the vendored repo's versions.** Colab ships torch
built against its own CUDA; pinning would trigger a multi-GB reinstall and a runtime
restart. Actual versions are recorded in the run manifest instead.

**Masked arithmetic rather than `np.where` in the distribution functionals.**
`np.where` evaluates both branches, so a zero-mass token computes `0 * -inf = NaN`
and then discards it. Same answer, but over a 128k vocabulary most tokens have zero
mass and the run fills with invalid-op warnings — burying the one real NaN that
would matter.

---

## 11. Which regime is each baseline actually in? **[build]**

Measured, not read off a name. `pilot/regime.py` recovers the τ a method is really
operating at by regressing its output on the log-space basis:

    adjusted = (1 - τ)·log p_pri + τ·log p_ctx + c

**Why bother.** The vendored `CoCoADecoding.get_tau()` returns `global_alpha` (0.5)
while the method actually operates at `alpha + gamma` (1.5). That is the difference
between interpolation and extrapolation — the entire distinction the paper is about —
and it is wrong in the direction that matters. We log `mean_tau` from `get_tau()`, so
without this we would have recorded 0.5 for a method sitting at 1.5.

**The centring is load-bearing.** `c` is never zero: normalising leaves a constant,
and CAD and AdaCAD combine *raw* logits, so their output differs from the log-space
form by both partition functions. An uncentred projection is biased by
`c·Σbasis/‖basis‖²`. Measured on a 2,000-token vocabulary, a true τ of 2.5 estimates
as **−0.30** — confident, plausible, and on the wrong side of τ=1. Mean-centring both
vectors fits the intercept implicitly and is exact to 1e-9 for every affine form
(power family, CAD at any α, CoCoA at any α and γ). Both facts are pinned as tests.

A large `affine_residual` means the method is not affine in log space and cannot be
summarised by one τ at all, which is a useful answer rather than a failure.

Stage 06 probes every baseline on a real conflict case before running it, prints the
effective τ next to the self-reported one, flags disagreements, and saves the table
to `test3b.json` under `regimes`.

### The CoCoA ambiguity, resolved empirically

The paper is internally inconsistent about CoCoA: its Eq. 5 gives a pure blend
`q ∝ p_ctx^λ · p_pri^(1−λ)` (interpolation, τ = λ), while Table 1 lists the row as
**CoCoA\*** with τ = λ + γ, classified as extrapolation. The asterisk suggests they
are tabulating a modified variant rather than CoCoA as published.

What the code actually does, verified numerically:

* `s_mix = α·log p_ctx + (1−α)·log p_pri + γ·(log p_ctx − log p_pri)`, which is
  algebraically `(α+γ)·log p_ctx + (1−α−γ)·log p_pri` — matching the power family at
  τ = α + γ to 7e-15.
* Repo defaults are `α=0.5, γ=1.0`, so **τ = 1.5: extrapolation, i.e. Table 1's
  CoCoA\*, not Eq. 5.**
* **γ is exposed** as a constructor parameter, so `gamma=0.0` recovers Eq. 5 exactly
  at τ = α = 0.5. Surfaced as `config.COCOA_GAMMA`.
* The `lambda_pm=100.0` median/MAD z-score term is a *monotone* transform of
  `s_mix`, so it preserves the entire ranking and is a **no-op for greedy decoding**
  (verified: argsort identical, nothing clamped at the ±100 bound). It matters only
  for sampling.

**Decision: run the repo's defaults (τ = 1.5) so our numbers match their table, and
report the effective τ alongside every baseline so the reading is unambiguous.** A
reviewer asking "which CoCoA is this?" is answered by `regimes` in `test3b.json`
rather than by our word.

Note this partly vindicates the argument that blending cannot exceed `p_ctx`: if
CoCoA as *published* is Eq. 5, it is interpolation and is so bounded. What cannot be
claimed is novelty — it is Corollary 5 of arXiv 2606.10298. **Still open:** read the
original CoCoA paper's own equation and record which version each side is comparing
against. Reviewers will ask.

---

## 12. Results to be recorded here after running — **[open]**

- [ ] Ambiguous fraction from stage 01, and whether 6/8 and 0/8 held up (§1.6).
- [ ] Agreement rate between our prior labels and TriState-Bench's GAPS labels.
- [ ] `lens_check.json`: which convention the lens found, and the reconstruction error.
- [ ] First-token-usable fraction, and whether Test 1 changed when restricted to it.
- [ ] Whether the vendored metrics imported, or the fallback was used (§1.7).
- [ ] The three tuned oracle τ constants.
- [ ] Measured two-pass overhead, batched and serial — the answer to the spec's "is
      2× real?" question.
