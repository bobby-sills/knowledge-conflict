"""Pilot study: internal knowledge as arbiter in knowledge conflict.

Logic lives in these modules; notebooks are for plots and narration. See
PILOT_SPEC.md for what is being tested and RESULTS.md for what came back.

Submodules
----------
config        every tunable and every kill-criterion threshold
io_utils      crash-tolerant JSONL / memmap IO, manifests, the report-split lock
splits        by-entity train / layer / report assignment
factset       PopQA loading, distractor construction, TriState-Bench loading
documents     faithful and corrupted document templates, prompt assembly
model         model loading and batched forward passes
lens          the logit lens, with its self-check against real logits
prior         Test 0 closed-book screening and state labelling
scoring       Test 1 internal/external candidate scores and knowledge scores
signals       Test 2 predictors: three internal, seven external
stats         AUC, bootstrap CIs, error-correlation matrix
reachability  Test 3a pairwise reversal thresholds
powerfamily   Test 3b power-family generation, tau sweep, oracle routing
permutation   Test 4 control
vendor        the published baselines and their metrics, pinned
figures       every plot named in the spec's Report sections
timing        two-pass vs single-pass wall-clock overhead
"""

__version__ = "0.1.0"
