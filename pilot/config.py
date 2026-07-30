"""Central configuration. Every tunable the spec left open lives here and only here.

Anything in this file that the spec did not determine must have a matching entry
in DECISIONS.md. Kill-criterion thresholds are declared here too, so they are
fixed in code before any result exists (Guardrail 1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def default_project_dir() -> Path:
    """Google Drive when mounted, else a local directory.

    Colab sessions die; Drive survives them. The whole point of resolving this
    once, here, is that no other module ever hardcodes a path.
    """
    env = os.environ.get("KC_PROJECT_DIR")
    if env:
        return Path(env)
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        return drive / "knowledge-conflict-pilot"
    return Path(__file__).resolve().parent.parent / "run"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def results(self) -> Path: return self.root / "results"

    @property
    def figures(self) -> Path: return self.root / "figures"

    @property
    def vendor(self) -> Path: return self.root / "vendor"

    @property
    def cache(self) -> Path: return self.root / "cache"

    def mkdirs(self) -> "Paths":
        for p in (self.results, self.figures, self.vendor, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self

    # ---- individual artefacts (stage -> file) ---------------------------- #
    @property
    def factset(self) -> Path: return self.results / "factset.jsonl"

    @property
    def prior(self) -> Path: return self.results / "prior_samples.jsonl"

    @property
    def states(self) -> Path: return self.results / "states.jsonl"

    @property
    def capture(self) -> Path: return self.results / "capture.jsonl"

    @property
    def resid(self) -> Path: return self.results / "resid.npy"

    @property
    def resid_index(self) -> Path: return self.results / "resid_index.parquet"

    @property
    def manifest(self) -> Path: return self.results / "manifest.jsonl"

    @property
    def report_lock(self) -> Path: return self.results / "REPORT_SPLIT.lock"

    def test_dir(self, name: str) -> Path:
        d = self.results / name
        d.mkdir(parents=True, exist_ok=True)
        return d


PATHS = Paths(default_project_dir())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DTYPE = "bfloat16"

# Prompt style.
#   "chat" — Llama-3 chat template, correct for an Instruct checkpoint.
#   "raw"  — the GRD repo's bare "Question: ...\nAnswer:" format.
# We capture under "chat" (it is what the model was trained for) and switch to
# "raw" when reproducing the vendored baselines so our numbers line up with
# theirs. See DECISIONS.md ("prompt format").
PROMPT_STYLE = "chat"

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer with the shortest possible span of text and nothing else. "
    "Do not explain, do not use a full sentence."
)

CTX_INSTRUCTION = "Using only the reference passage above, answer the question."


# --------------------------------------------------------------------------- #
# Test 0 — fact set and prior screening
# --------------------------------------------------------------------------- #

FACTSET_NAME = "popqa"          # "popqa" | "tristate" (see factset.py)
TARGET_FACTS = 2000             # spec: "a few hundred to ~2,000 facts"
POPQA_HF_ID = "akariasai/PopQA"
POPQA_SPLIT = "test"

N_DISTRACTORS = 8               # candidate set = 1 gold + N distractors
DISTRACTOR_POP_TOLERANCE = 1.0  # log10 pageview band for popularity matching
DISTRACTOR_SEED = 20260729

PRIOR_N_SAMPLES = 8
PRIOR_TEMPERATURE = 0.7
PRIOR_TOP_P = 0.9
PRIOR_MAX_NEW_TOKENS = 24
PRIOR_CORRECT_MIN = 6           # >= 6/8 hits  -> prior correct
PRIOR_WRONG_MAX = 0             # == 0/8 hits  -> prior wrong; else ambiguous
PRIOR_SEED = 1234

# Popularity binning for the Test 0 / Test 2 popularity reports.
POP_BIN_EDGES = (0, 100, 1_000, 10_000, 100_000, float("inf"))


# --------------------------------------------------------------------------- #
# Splits — by entity, assigned once, in the first stage
# --------------------------------------------------------------------------- #

SPLIT_FRACTIONS = {"train": 0.40, "layer": 0.30, "report": 0.30}
SPLIT_SALT = "kc-pilot-v1"
DEV_SPLITS = ("train", "layer")   # everything Tests 0-4 may look at


# --------------------------------------------------------------------------- #
# Capture / scoring
# --------------------------------------------------------------------------- #

TOPK_SAVE = 64                  # top-k probs persisted per pass for re-analysis
CAPTURE_BATCH_SIZE = 8
SAVE_RESIDUALS = True
RENYI_ALPHA = 2.0               # CoCoA's D2

# Candidate scoring granularity.
#   "first_token"  — distribution over the first answer token (comparable to the
#                    logit lens, which only ever sees one position)
#   "full_meanlp"  — length-normalised teacher-forced log-prob of the whole span
# The first is primary; the second is logged as a secondary external score.
PRIMARY_SCORE = "first_token"


# --------------------------------------------------------------------------- #
# Test 3 — power family
# --------------------------------------------------------------------------- #

TAU_GRID = tuple(round(0.05 * i, 3) for i in range(0, 51))   # 0.00 .. 2.50
GEN_MAX_NEW_TOKENS = 24
ORACLE_TAU_INIT = {"correction": 1.5, "resistance": 0.3, "agreement": 1.0}

VENDOR_REPO = "https://github.com/keith-Jiang/Gated-Reversal-Decoding"

# Pinned to the last commit that contains `methods/arr.py` — **not** to HEAD.
#
# ARR (Adaptive Regime Routing) is the method arXiv 2606.10298 actually publishes;
# its abstract credits ARR with lifting resistance EM "from below 6 to 16--33". On
# 2026-07-29 the authors replaced `arr.py` with `grd.py` (Gated Reversal Decoding)
# and renamed the repo, so HEAD no longer contains the paper's method at all. GRD is
# a *different algorithm*, not a rename: ARR is stateless and routes per step
# between extrapolation and interpolation, while GRD is stateful and never
# extrapolates. Comparing against HEAD would compare against an unpublished
# successor. See DECISIONS.md §1.2.
VENDOR_COMMIT = "320d88bc"          # last commit containing methods/arr.py
VENDOR_COMMIT_DATE = "2026-07-09"
VENDOR_HEAD_WITH_GRD = "581a4d59"   # recorded, deliberately not used

# `greedy` and `greedy_no_ctx` are omitted: they are exactly PowerFamily(1.0) and
# PowerFamily(0.0), which the tau sweep already covers, so vendoring them would add
# two more code paths computing numbers we already have.
VENDOR_METHODS = ("cad", "adacad", "cocoa", "coiecd", "arr")

# The paper's reported range for ARR's resistance EM. Not a threshold we invented —
# a reproduction target. If our ARR run lands far outside it, we have the wrong
# method or a broken config, whatever the class is called, and Test 3's kill
# criterion is measured against a number that does not mean what we think.
ARR_RESISTANCE_EM_TARGET = (0.16, 0.33)

# CoCoA's effective exponent is alpha + gamma, so the repo's defaults put it at
# tau = 1.5 (extrapolation) — Table 1's `CoCoA*`, not Eq. 5's pure blend. Setting
# gamma = 0.0 recovers Eq. 5 at tau = alpha. We run the repo's default so our
# numbers match their table, and report both readings. See DECISIONS.md §11.
COCOA_ALPHA = 0.5
COCOA_GAMMA = 1.0


# --------------------------------------------------------------------------- #
# Test 4
# --------------------------------------------------------------------------- #

PERMUTATION_N_SHUFFLES = 20
PERMUTATION_SEED = 777


# --------------------------------------------------------------------------- #
# Kill criteria — fixed in advance. Do not edit after seeing a result.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KillCriteria:
    # Test 1
    min_divergence_rate: float = 0.10
    # "internal must beat external on the divergent subset": strict inequality on
    # accuracy among facts where the two top-1 answers differ.
    divergent_subset_internal_must_win: bool = True
    # Test 2
    min_auc_margin: float = 0.05
    require_non_overlapping_ci: bool = True
    # Test 3
    min_oracle_gain_over_best_baseline_em: float = 0.05   # 5 EM points
    # Test 4
    max_permuted_fraction_of_real_gain: float = 0.50


KILL = KillCriteria()


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 99


def snapshot() -> dict:
    """Everything in this module that a result depends on, for the manifest."""
    return {
        "model_id": MODEL_ID,
        "dtype": DTYPE,
        "prompt_style": PROMPT_STYLE,
        "factset": FACTSET_NAME,
        "target_facts": TARGET_FACTS,
        "n_distractors": N_DISTRACTORS,
        "prior": {
            "n_samples": PRIOR_N_SAMPLES,
            "temperature": PRIOR_TEMPERATURE,
            "top_p": PRIOR_TOP_P,
            "correct_min": PRIOR_CORRECT_MIN,
            "wrong_max": PRIOR_WRONG_MAX,
        },
        "splits": SPLIT_FRACTIONS,
        "split_salt": SPLIT_SALT,
        "topk_save": TOPK_SAVE,
        "primary_score": PRIMARY_SCORE,
        "renyi_alpha": RENYI_ALPHA,
        "tau_grid": [TAU_GRID[0], TAU_GRID[-1], len(TAU_GRID)],
        "vendor_commit": VENDOR_COMMIT,
        "vendor_methods": list(VENDOR_METHODS),
        "cocoa": {"alpha": COCOA_ALPHA, "gamma": COCOA_GAMMA,
                  "effective_tau": COCOA_ALPHA + COCOA_GAMMA},
        "kill": asdict(KILL),
    }
