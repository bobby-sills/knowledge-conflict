# RESULTS

**Status: pipeline built, not yet run.** No model has been executed. Every number
below is a placeholder marked `TBD`, and the kill-criterion column is unfilled by
design — the thresholds are fixed in `pilot/config.py:KillCriteria` and were
written before any result existed.

Write this file **as you go**, not at the end. One section per test, filled in as
each stage completes, with the plots inlined from `figures/`.

Model: `meta-llama/Meta-Llama-3-8B-Instruct`, bf16.
Fact set: PopQA, target 2,000 facts (see DECISIONS.md §2).
Vendored baselines: `keith-Jiang/Gated-Reversal-Decoding` @ `581a4d59`.

---

## Kill-criterion summary

Fill this in first, from the top, as each stage runs. If a criterion fires: **stop,
write it up, report.** Do not adjust the threshold, do not search for a variant that
passes, do not run the next test. A pilot that kills the project in two weeks has
succeeded.

| Test | Criterion (fixed in advance) | Result | Fired? |
|---|---|---|---|
| 0 | none — diagnostic | TBD | n/a |
| 1 | divergence ≥ 10% **and** internal beats external on the divergent subset | TBD | TBD |
| 2 | best internal ≥ best external + 0.05 AUC, non-overlapping CIs | TBD | TBD |
| 3 | oracle 3-way routing ≥ best baseline + 0.05 EM | TBD | TBD |
| 4 | shuffled signal recovers ≤ 50% of the real gain | TBD | TBD |

**Instrument check (must pass before Test 1 is believed):**

| Check | Expected | Result |
|---|---|---|
| Logit lens reproduces the model's final-layer logits | max abs err ≤ 4 ULP of the compute dtype | **PASS** — 0.0625 = **0.50 ULP** of bf16 (tol 0.5) |
| The two lens conventions are cleanly separated | ≥ 20× | **PASS** — **89.5×** (0.0625 vs 5.594) |
| Lens reproduces the model's token ranking | top-1 and top-8 identical | **PASS** — top-64 identical *in order* |
| Lens convention detected | `post_norm` or `pre_norm` | **`post_norm`** (last hidden state is already normed) |
| Final-layer lens == external scores | identical to 1e-3 | TBD |
| No entity leaks across splits | 0 leaks | TBD |
| **ARR reproduces the paper's resistance EM** | **0.16 – 0.33** | **TBD** |
| Vendored metrics imported (not the fallback) | True | TBD |

The ARR row is not optional. Test 3's kill criterion is measured *relative to* the
best baseline, so a misconfigured comparator moves the gate without announcing it.
If ARR lands outside 0.16–0.33, resolve that before reading Test 3 at all — check
the prompt style, the fact set, and the measured-τ table below.

### Which regime is each baseline actually in?

Measured by `pilot/regime.py`, not read off names or `get_tau()`. Filled in by stage
06 (`test3b.json` → `regimes`).

| Method | effective τ | self-reported τ | regime | agree? |
|---|---|---|---|---|
| CAD | TBD | 1.5 | extrapolation | TBD |
| AdaCAD | TBD | 1 + JSD | extrapolation | TBD |
| CoCoA | TBD | 0.5 | **expect 1.5** | **expect NO** |
| COIECD | TBD | TBD | TBD | TBD |
| ARR | TBD | per-step ±s | **routes across τ=1** | TBD |

CoCoA's row is the known discrepancy: the repo's `get_tau()` returns `alpha` (0.5)
while the method operates at `alpha + gamma` (1.5). That is Table 1's `CoCoA*`
extrapolation variant, not Eq. 5's pure blend — recorded in DECISIONS.md §11, with
`gamma=0.0` available to recover Eq. 5 if a reviewer asks for that comparison.

---

## Test 0 — Does the resistance state exist in useful volume?

*Stage: `python -m pilot.cli prior`. Artefacts: `results/test0/summary.json`,
`results/prior_samples.jsonl`, `results/states.jsonl`.*

### Prior calibration

600 facts screened, 8 closed-book samples each at T=0.7.

| | Count | Fraction |
|---|---|---|
| prior correct (≥ 6/8) | 141 | 23.5% |
| prior wrong (0/8) | 401 | 66.8% |
| ambiguous (1-5/8) | 58 | 9.7% |

**Ambiguous fraction: 9.7%.** Well under the 50% flag, so the 6/8 and 0/8 thresholds
hold up for this model and needed no adjustment. The distribution is strongly
bimodal — the model either knows a PopQA long-tail fact or has no idea, and rarely
sits in between. That is a convenient property and not one we assumed.

The asymmetry is the other way from what the pilot would prefer: the prior is *wrong*
on two thirds of the set. PopQA is deliberately long-tail, so this is expected, but it
means correction cases are abundant and resistance cases are the scarce resource.

### Conflict states

683 conflict cases from 600 facts. Prior-correct facts yield two cases each
(resistance and agreement); prior-wrong facts yield one (correction); ambiguous facts
are excluded.

| State | Prior | Context | Cases |
|---|---|---|---|
| correction | wrong | faithful | 401 |
| resistance | correct | corrupted | 141 |
| agreement | correct | faithful | 141 |

**141 resistance cases** clears the 100-case floor, but only just. Test 2's AUC on the
resistance side will have wide intervals, so report the width, not the point. If a
result comes back promising, the cheapest way to tighten it is more facts — and
because the prior is wrong 67% of the time, scaling the fact set buys resistance cases
at roughly one per four facts screened.

![state counts](figures/test0_states.png)

### Entity popularity by state

![popularity by state](figures/test0_popularity.png)

Median log10 subject pageviews: correction **2.84**, resistance **3.73**, agreement
**3.73**.

**This confound is real, and it is large.** Resistance and agreement share a median by
construction — both are built from the same 141 prior-correct facts — so the
comparison that matters is correction (2.84) against resistance (3.73): a gap of
**0.89 decades**, nearly an order of magnitude in pageviews.

That is not a nuisance, it is close to a tautology: the model knows popular entities,
and "the model knows it" is exactly what separates the prior-correct states from the
prior-wrong one. So log-popularity is expected to be a *strong* predictor of Test 2's
binary while carrying no information whatsoever about the model's internal state.

Two consequences for how Test 2 must be read:

1. **Beating chance is meaningless here.** The bar an internal signal has to clear is
   log-popularity's AUC, not 0.5. `signals.py` carries log-popularity as one of the
   seven external predictors precisely so this shows up in the same table.
2. **A high-AUC internal signal is not yet evidence.** If an internal signal and
   log-popularity have correlated errors, the internal signal may be reading
   popularity off the residual stream rather than reading the conflict. The
   error-correlation matrix is the check, and on this fact set it is load-bearing
   rather than a formality.

### Cross-check against TriState-Bench

The vendored repo ships prior-screened states for this exact model, produced by a
different procedure (GAPS). Agreement with our labels: TBD.

### Limitation, stated out loud

The documents are **synthetic template passages**, not retrieved text. They are
shorter, more assertive and single-claim, which inflates the apparent authority of
the context and removes every partial or hedged conflict. Acceptable for a pilot;
not acceptable for the paper. See DECISIONS.md §1.5 for the replacement candidates
(ClashEval / ConflictQA / ConFiQA).

Whether the resistance state is rare under *realistic* retrieval is not answered
here and cannot be — it is constructed here, by design. **If it turns out to be
constructible only synthetically, that changes how the paper must be positioned**,
and belongs in this section prominently.

---

## Test 1 — Do internal and external answers diverge?

*Stage: `python -m pilot.cli test1`. Artefacts: `results/test1/test1.json`.*
*Layer chosen on the `layer` split; numbers reported on `train`.*

### Instrument check

Ran before capture. **Result: PASS** — convention `post_norm`, reconstruction error
0.0625 (half a bf16 ULP, i.e. the arithmetic floor), conventions separated by 89.5×,
top-64 token order reproduced exactly. See DECISIONS.md §10 for why a mis-detected
lens produces plausible, wrong numbers rather than an error.

The first attempt at this check *failed*, on a tolerance of 2e-2 that was below what
bf16 can represent at these logit magnitudes. That was a fault in the check, not in
the lens; DECISIONS.md §10 records the diagnosis and the replacement criterion. Worth
noting for the write-up: the guardrail behaved correctly in the sense that mattered —
it stopped the pipeline before any data was captured, rather than letting a
half-verified instrument through.

### Knowledge scores

| | Score |
|---|---|
| internal (logit lens, best layer) | TBD |
| external (output probabilities) | TBD |
| gap | TBD |

Best layer: **TBD** of 32. (Expect mid-to-late.)

![knowledge by layer](figures/test1_knowledge_by_layer.png)

### Divergence

| | Value |
|---|---|
| internal top-1 ≠ external top-1 | TBD |
| n diverged | TBD |
| on divergent: internal correct, external wrong | TBD |
| on divergent: external correct, internal wrong | TBD |
| on divergent: both wrong | TBD |

![internal vs external](figures/test1_scatter.png)

*Candidate Figure 1. Each point is a fact; colour is conflict state; the diagonal
separates facts where the internals rank the gold better than the output
distribution does.*

### First-token collisions

Usable (collision-free) facts: TBD. Test 1 restricted to those: TBD. If the
restricted and unrestricted numbers disagree materially, collisions are driving the
result and the restricted number is the one to believe.

### Kill criterion

Divergence ≥ 10% **and** internal beating external on the divergent subset:
**TBD**.

---

## Test 2 — Does the internal signal predict the *decision*?

*Stage: `python -m pilot.cli test2`. Artefacts: `results/test2/test2.json`.*
*Directions and thresholds fitted on `train`; AUCs reported on `layer`.*
***This is the real gate.***

Binary: on correction and resistance cases only, **should we resist?** Ground truth
from Test 0's state labels. Every signal is oriented so larger means "resist".

| Signal | Kind | AUC | 95% CI |
|---|---|---|---|
| internal_knowledge | internal | TBD | TBD |
| internal_margin | internal | TBD | TBD |
| trajectory_stability | internal | TBD | TBD |
| prior_entropy | external | TBD | TBD |
| prior_max | external | TBD | TBD |
| entropy_gap (CoCoA) | external | TBD | TBD |
| jsd (AdaCAD) | external | TBD | TBD |
| renyi (CoCoA) | external | TBD | TBD |
| log_popularity | **control** | TBD | TBD |
| self_consistency | external | TBD | TBD |

![AUC table](figures/test2_auc.png)

Best internal: TBD. Best external: TBD. Margin: TBD. CIs disjoint: TBD.
Paired difference (bootstrap, shared resamples): TBD.

**Read log_popularity first.** If it matches the best internal signal, the internal
signal is a frequency detector and the framing of the whole project is wrong.

### Error correlation

![error correlation](figures/test2_error_correlation.png)

**This matters even if we lose.** If the internal signal's errors are uncorrelated
with the external signals' errors, it is *complementary* rather than superior, and
the project becomes an ensemble method — weaker, still viable. Report that
explicitly rather than reporting a bare failure. `--` cells are undefined (a
predictor with constant errors), not zero.

### Permutation null (cheap control, run on every signal)

Shuffled AUCs should sit at 0.5. Any signal whose shuffled AUC does not is
evidence of a broken analysis, not a finding: TBD.

### Kill criterion

Best internal ≥ best external + 0.05 AUC with non-overlapping CIs: **TBD**.

---

## Test 3 — Is there anything to steer toward?

### 3a. Analytic reachability

*Stage: `python -m pilot.cli test3a`. Artefacts: `results/test3a/test3a.json`.*

τ*(a,b) = −ℓ_pri(a,b) / (ℓ_ctx(a,b) − ℓ_pri(a,b)) — the τ at which the model's
preference between a and b flips.

| State | median τ* | fraction in (0,1) | undefined |
|---|---|---|---|
| correction | TBD | TBD | TBD |
| resistance | TBD | TBD | TBD |
| agreement | TBD | TBD | TBD |

![tau star](figures/test3a_tau_star.png)

Theory predicts τ* ∈ (0,1) for resistance, meaning interpolation is sufficient and
nothing exotic is needed. Verified: **TBD**. If fewer than half of resistance cases
land in (0,1), the prediction does not hold here — report it, because it changes
what the eventual method has to do.

### 3b. Oracle ceiling

*Stage: `python -m pilot.cli test3b`. Artefacts: `results/test3b/test3b.json`.*

![tau sweep](figures/test3b_tau_sweep.png)

*Per-state EM across the power family. The crossing of the correction and
resistance curves is the regime asymmetry: no single τ tops both.*

![trade-off](figures/test3b_tradeoff.png)

*The fixed-τ frontier. Any method above and to the right of it is doing something a
global τ cannot.*

| Method | Overall EM | Correction | Resistance | Agreement |
|---|---|---|---|---|
| τ=1.0 (pure context, ≡ greedy) | TBD | TBD | TBD | TBD |
| τ=0.0 (pure prior, ≡ greedy_no_ctx) | TBD | TBD | TBD | TBD |
| CAD | TBD | TBD | TBD | TBD |
| AdaCAD | TBD | TBD | TBD | TBD |
| CoCoA* (τ = α+γ = 1.5) | TBD | TBD | TBD | TBD |
| COIECD | TBD | TBD | TBD | TBD |
| **ARR** (the published comparator) | TBD | TBD | TBD | TBD |
| best fixed τ (τ = TBD) | TBD | TBD | TBD | TBD |
| **oracle 3-way** | TBD | TBD | TBD | TBD |

Tuned oracle constants: correction τ=TBD, resistance τ=TBD, agreement τ=TBD.

The comparator is **ARR** (Adaptive Regime Routing), the method arXiv 2606.10298
publishes, from the pinned commit `320d88bc`. Note that the repo's HEAD replaced ARR
with a different algorithm (GRD) on 2026-07-29; we deliberately do not use HEAD.
DECISIONS.md §1.2 has the full account.

### Kill criterion

Oracle ≥ best baseline + 0.05 EM: **TBD**. A high overall ceiling with a flat
resistance gain is still a dead end for this project, so read `resistance_gain`
alongside the headline.

---

## Test 4 — Permutation control

*Stage: `python -m pilot.cli test4`. Artefacts: `results/test4/test4.json`.*

Signal tested: TBD (the Test 2 winner). Threshold fitted on `train`.

### Signal level (free)

Real AUC TBD; shuffled mean TBD (should be ≈ 0.5); p = TBD.

### Decoding level

Gains measured against the **best fixed τ**, not against τ=1 — against τ=1 a pure
global rescale would look like a win and pass a control designed to catch exactly
that.

| | Overall EM | Gain over best fixed τ |
|---|---|---|
| routed by the real signal | TBD | TBD |
| routed by shuffled signal (mean of N) | TBD | TBD |
| routed by shuffled signal (max) | TBD | TBD |

![permutation control](figures/test4_permutation.png)

Recovered fraction: TBD. Within-state shuffle (stricter): TBD.

### Kill criterion

Shuffled runs recover ≤ 50% of the real gain: **TBD**.

---

## Two-pass overhead

*Stage: `python -m pilot.cli timing`. Artefacts: `results/timing/timing.json`.*

The literature frames contrastive decoding as "2× cost". Measured here because it
was nearly free to answer.

| Configuration | ms/sequence | ×  vs plain greedy | ms/token |
|---|---|---|---|
| single (plain greedy) | TBD | 1.00 | TBD |
| single, batch of 2 | TBD | TBD | TBD |
| two-pass, batched | TBD | TBD | TBD |
| two-pass, serial | TBD | TBD | TBD |

Conclusion: TBD.

---

## What we can answer, with numbers

The five questions the pilot exists to answer:

1. **Do internal and external knowledge measurements diverge on this model? By how
   much?** — TBD
2. **When they diverge, is the internal one right?** — TBD
3. **Does an internal signal predict the resist-or-correct decision better than the
   six output-distribution signals the field currently uses?** — TBD
4. **What is the ceiling if the signal were perfect?** — TBD
5. **Is any apparent gain per-question adaptivity, or a global rescale?** — TBD

**Recommendation: TBD.** If (1)-(3) come back positive and (4) shows headroom and
(5) survives, build the full project. Otherwise we have learned something real and
saved four months — state that plainly, without editorialising it into a maybe.
