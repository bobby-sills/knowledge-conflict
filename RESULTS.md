# RESULTS

## 🔴 Verdict: no-go. Test 1's kill criterion fired.

**Internal and external answers diverge on half the fact set — but where they diverge,
the output distribution is right more often than the internal state (24.0% vs 19.8%).**
The premise of the project is that the residual stream is the better arbiter. On
Llama-3-8B-Instruct over 600 PopQA facts, it is the worse one.

Tests 2, 3 and 4 were not run. No threshold was adjusted and no variant was tried
after the result was seen. The `report` split is still locked and unexamined.

Both instrument checks passed *before* the result was read, so this is a measurement
rather than an artefact: the logit lens reconstructs the model's logits to half a bf16
ULP with the wrong convention 89.5× worse, and the final-layer lens matches the
external scores to 5.0e-7.

Run: 2026-07-29, A100-40GB, ~5 min for capture. Details below; the reasoning behind
every judgement call is in [DECISIONS.md](DECISIONS.md).

Model: `meta-llama/Meta-Llama-3-8B-Instruct`, bf16.
Fact set: PopQA, **600 facts** → 683 conflict cases (see DECISIONS.md §2).
Vendored baselines: `keith-Jiang/Gated-Reversal-Decoding` @ `320d88bc` — never
reached, since Test 3 did not run.

---

## Kill-criterion summary

Fill this in first, from the top, as each stage runs. If a criterion fires: **stop,
write it up, report.** Do not adjust the threshold, do not search for a variant that
passes, do not run the next test. A pilot that kills the project in two weeks has
succeeded.

| Test | Criterion (fixed in advance) | Result | Fired? |
|---|---|---|---|
| 0 | none — diagnostic | 683 cases; 141 resistance; 9.7% ambiguous | n/a |
| 1 | divergence ≥ 10% **and** internal beats external on the divergent subset | divergence **50.2%** ✔; on divergent internal **0.198** vs external **0.240** ✘ | **🔴 YES** |
| 2 | best internal ≥ best external + 0.05 AUC, non-overlapping CIs | not run | — |
| 3 | oracle 3-way routing ≥ best baseline + 0.05 EM | not run | — |
| 4 | shuffled signal recovers ≤ 50% of the real gain | not run | — |

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

A second validity check, also required before believing anything below: the
final-layer lens *is* the output distribution, so the internal score at layer 32 must
equal the external score. **Max |lens(L32) − external| = 5.0e-7** across all 542 facts
(threshold 1e-3), and all 542 rank the candidates identically at the final layer. The
internal-vs-external comparison therefore measures what it claims to.

### Knowledge scores

Layer chosen on the `layer` split, reported on `train` (n = 241 facts).

| | Score |
|---|---|
| internal (logit lens, layer 28) | 0.7443 |
| external (output probabilities) | 0.7759 |
| gap | **−0.0316** |

Best layer: **28** of 32 — mid-to-late, as expected.

![knowledge by layer](figures/test1_knowledge_by_layer.png)

**The layer-28 peak does not replicate out of sample, and that is the whole story.**
The per-layer curve on the selection split is flat at ~0.48 through layer 14, then
climbs monotonically to a peak of 0.7844 at layer 28 and falls back to 0.7631 at layer
32. Because layer 32 is by construction the external score, on the *selection* split
the best internal layer beats external by +0.021. On `train` the same layer is behind
by −0.032:

| Split | Layer 28 (chosen) | Layer 32 (= external) | Internal − external |
|---|---|---|---|
| `layer` (selection) | 0.7844 | 0.7631 | **+0.021** |
| `train` (report) | 0.7443 | 0.7759 | **−0.032** |

A ~5-point swing between splits at n ≈ 241 is what selection noise looks like. The
mid-layer advantage is an artefact of choosing the layer that maximised it, and the
held-out split is the number that counts. Had the layer been selected on `train`, this
row would have read positive and meant nothing — which is exactly why the split exists.

### Divergence

| | Value |
|---|---|
| internal top-1 ≠ external top-1 | **50.2%** |
| n diverged | 121 of 241 |
| on divergent: internal correct | 19.8% |
| on divergent: external correct | **24.0%** |
| on divergent: both wrong | 56.2% |

![internal vs external](figures/test1_scatter.png)

*Candidate Figure 1. Each point is a fact; colour is conflict state; the diagonal
separates facts where the internals rank the gold better than the output
distribution does.*

**The first clause passes handsomely and the second fails.** Internal and external
disagree on half the fact set — five times the 10% floor — so the phenomenon the
project is premised on is unambiguously real. What fails is the claim that follows it:
where they disagree, the output distribution is right *more* often than the internal
state, 24.0% against 19.8%.

The third row is the one that explains the other two. On 56.2% of divergent facts
**neither** signal is right. Divergence is not concentrated where the model quietly
knows the answer and the output layer garbles it; it is concentrated where the model
does not know the answer at all, and the two readings disagree because both are noise.
That is a coherent alternative account of why divergence is high, and it does not
support an arbiter.

### First-token collisions

Usable (collision-free) facts: **511/542 (94.3%)** at capture; **230 of the 241**
reported facts. Restricting Test 1 to them changes nothing material:

| | All 241 | Usable 230 |
|---|---|---|
| gap | −0.0316 | −0.0285 |
| divergence rate | 50.2% | 49.1% |
| on divergent: internal | 19.8% | 21.2% |
| on divergent: external | 24.0% | 25.7% |

The kill fires on both. Collisions are not driving the result.

### Kill criterion

Divergence ≥ 10% **and** internal beating external on the divergent subset:

> ### 🔴 FIRED — the pilot stops here.
>
> - Divergence 50.2% ≥ 10% — **passes**.
> - On the divergent subset, internal 0.198 vs external 0.240 — **fails**.
>
> `results/test1/test1.json` → `kill.fired = true`.

Per the spec and `pilot/config.py`, Tests 2, 3 and 4 were **not run**. No threshold was
adjusted, no alternative layer, score, or prompt variant was tried after seeing this
result. The `report` split remains locked and unexamined.

Both instrument checks passed before this was read (lens at 0.50 ULP with 89.5×
separation; final-layer lens matching external to 5.0e-7), so the null is a measurement
and not an artefact.

---

## Tests 2–4 — not run

Test 1's kill criterion fired, so the pipeline stopped. Stages 04–08 were never
executed and every number below remains TBD by design, not by omission. They are left
in place because the code is written and tested, and a revised premise (see "What we
can answer, with numbers") could make them worth running.

One measurement from Test 0 is worth carrying forward to whoever runs Test 2 next: the
0.89-decade popularity gap between correction and resistance means log-popularity will
post a strong AUC on that binary by itself. The bar there was never 0.5.

---

## Test 2 — Does the internal signal predict the *decision*? — **NOT RUN**

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
   much?** — **Yes, a lot. 50.2% of facts** have a different top-1 candidate under the
   layer-28 lens than under the output distribution. Five times the 10% floor.
2. **When they diverge, is the internal one right?** — **No.** Internal 19.8%, external
   24.0%. And on 56.2% of divergent facts neither is right, which is the more
   informative number: divergence tracks *ignorance*, not a suppressed correct answer.
3. **Does an internal signal predict the resist-or-correct decision better than the
   output-distribution signals the field uses?** — **Not run.** Test 1 is the
   precondition; there is no reason to price an arbiter that loses to what it replaces.
4. **What is the ceiling if the signal were perfect?** — Not run.
5. **Is any apparent gain per-question adaptivity, or a global rescale?** — Not run.

### Recommendation: do not build the full project as specified.

The specific claim the four-month project rests on — that the residual stream holds a
better answer than the output layer, which contrastive decoding then fails to exploit —
is false on this model and fact set, in the direction that matters. The output
distribution is not a degraded view of a cleaner internal state; on the facts where the
two disagree it is the *better* view.

Stated plainly and without hedging it into a maybe: **this saves roughly four months.**

**What the pilot did *not* rule out**, stated precisely so it is not overread:

- That divergence is real and large (50.2%). That part of the premise held.
- That a *trained probe* could beat the output distribution. We tested the untrained
  logit lens, which the spec chose deliberately as the cheap first look. A probe
  trained on the residual stream is a different instrument and this is not evidence
  about it — though note the ceiling it would have to clear: on 56% of divergent facts
  the gold answer is not top-1 in either reading, so the headroom a probe is competing
  for is smaller than the 50% divergence rate suggests.
- That the result holds beyond Llama-3-8B-Instruct, PopQA long-tail facts, and
  synthetic single-claim documents. All three are narrow, and the documents in
  particular are the limitation flagged in Test 0.
- That first-token scoring is the right granularity. It is what the lens can see, so
  the comparison is apples-to-apples, but a full-span internal score is not something
  a single-position lens can provide.

**If one thing were to be re-run before abandoning the direction**, it is Test 1 with a
trained linear probe in place of the logit lens, on the same captured residuals — the
capture is on disk, the `report` split is still clean, and it is hours rather than
months. That is a different experiment with a different premise, not a retry of this
one, and it should get its own kill criterion fixed in advance.
