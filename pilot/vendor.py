"""Bridge to the published baselines. We do not reimplement CAD/AdaCAD/CoCoA/
COIECD/ARR — the spec says use the authors' repo, and a reimplementation is a
reviewer's first target.

Repo: github.com/keith-Jiang/Gated-Reversal-Decoding (renamed from
`conflict-aware-decoding`, the name the spec cites), pinned at
config.VENDOR_COMMIT.

What we take:
  * `methods/*.py` — single-step logit combiners with a uniform interface
    (`get_next_token_logits(logits_ctx, logits_prior)`), which slots straight
    into our own generation loop.
  * `evaluation/utils.py` — SQuAD-style normalisation, `exact_match`,
    `substring_match`, `f1_score`. Using theirs, not ours, is what makes our EM
    numbers comparable to their table.
  * `data/TriState/Meta-Llama-3-8B-Instruct/` — prior-screened conflict states
    for our exact model.

What we do NOT take: their `inference.py` / shell harness, which is built around
a dual-GPU launcher and their own JSONL layout. We drive the method objects
directly.

**Why this pins an old commit.** ARR (Adaptive Regime Routing) is the method
arXiv 2606.10298 publishes. On 2026-07-29 the authors deleted `methods/arr.py`,
added `methods/grd.py` (Gated Reversal Decoding), and renamed the repo — so HEAD
does not contain the paper's method. GRD is a different algorithm, not a rename:

    ARR  stateless, per step:  tau = 1 + s  if p_ctx_max > p_pri_max else 1 - s
         with s = JSD/log2 clamped to [0,1]. Routes across the tau = 1 boundary.

    GRD  stateful: locks the trusted branch at the first conflict step, then uses
         tau = (1 - lambda) * tau_star with lambda = 0.75 and tau_star in (0,1),
         so tau <= 0.25. Never extrapolates.

We pin the last commit with ARR. See DECISIONS.md §1.2.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config

_REPO_DIRNAME = "Gated-Reversal-Decoding"


def ensure_repo(vendor_dir: Path | None = None, commit: str = config.VENDOR_COMMIT) -> Path:
    """Clone (or reuse) the pinned repo. Idempotent; safe after a dead session."""
    vendor_dir = Path(vendor_dir or config.PATHS.vendor)
    vendor_dir.mkdir(parents=True, exist_ok=True)
    root = vendor_dir / _REPO_DIRNAME

    if not (root / ".git").exists():
        print(f"[vendor] cloning {config.VENDOR_REPO} -> {root}")
        subprocess.run(["git", "clone", "--quiet", config.VENDOR_REPO, str(root)],
                       check=True)

    have = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if have != commit:
        print(f"[vendor] checking out {commit[:10]} (was {have[:10]})")
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin", commit],
                       check=False)
        subprocess.run(["git", "-C", str(root), "checkout", "--quiet", commit],
                       check=True)
    return root


def add_to_path(root: Path) -> Path:
    """Their modules import as `methods.base` / `evaluation.utils`, so the repo
    root itself has to be importable."""
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load(vendor_dir: Path | None = None) -> Path:
    return add_to_path(ensure_repo(vendor_dir))


# --------------------------------------------------------------------------- #
# Metrics — vendored, with a self-contained fallback
# --------------------------------------------------------------------------- #

def _fallback_normalise(s: str, remove_punctuation: bool = True) -> str:
    import re
    import string
    text = s.lower()
    if remove_punctuation:
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _metrics():
    """Their metric functions if the repo is available, ours otherwise.

    The fallback is a byte-for-byte port of their `normalize_answer`, so the two
    paths agree; it exists only so that Tests 0-2 (which need `substring_match`
    for prior screening) can run before the clone, e.g. in unit tests.
    """
    try:
        from evaluation.utils import (exact_match, f1_score,   # type: ignore
                                      normalize_answer, substring_match)
        return exact_match, substring_match, f1_score, normalize_answer, True
    except Exception:
        def exact_match(prediction: str, ground_truth: str) -> bool:
            return _fallback_normalise(prediction) == _fallback_normalise(ground_truth)

        def substring_match(ground_truth: str, prediction: str) -> bool:
            return _fallback_normalise(ground_truth) in _fallback_normalise(prediction)

        def f1_score(prediction: str, ground_truth: str) -> float:
            from collections import Counter
            pred = _fallback_normalise(prediction).split()
            gold = _fallback_normalise(ground_truth).split()
            common = Counter(pred) & Counter(gold)
            same = sum(common.values())
            if same == 0:
                return 0.0
            p, r = same / len(pred), same / len(gold)
            return 2 * p * r / (p + r)

        return exact_match, substring_match, f1_score, _fallback_normalise, False


def exact_match(prediction: str, ground_truth: str) -> bool:
    return _metrics()[0](prediction, ground_truth)


def substring_match(ground_truth: str, prediction: str) -> bool:
    return _metrics()[1](ground_truth, prediction)


def f1_score(prediction: str, ground_truth: str) -> float:
    return _metrics()[2](prediction, ground_truth)


def normalize_answer(s: str) -> str:
    return _metrics()[3](s)


def using_vendored_metrics() -> bool:
    """True when the metrics above came from the repo. Recorded in the manifest —
    a run scored with the fallback is not comparable to their published table."""
    return _metrics()[4]


def em_any(prediction: str, golds) -> bool:
    return any(exact_match(prediction, g) for g in golds)


def substring_any(prediction: str, golds) -> bool:
    return any(substring_match(g, prediction) for g in golds)


# --------------------------------------------------------------------------- #
# Baseline decoding methods
# --------------------------------------------------------------------------- #

def build_methods(names=config.VENDOR_METHODS, **kwargs) -> dict:
    """Instantiate the vendored method objects.

    Returns name -> object, each exposing
    `get_next_token_logits(logits_ctx, logits_prior)`. `powerfamily.generate` also
    handles the stateful `select_next_token` / `reset` shape, which is what GRD uses
    at HEAD, but no method here needs it.

    `greedy` and `greedy_no_ctx` are deliberately absent: they are PowerFamily(1.0)
    and PowerFamily(0.0), already computed by the tau sweep.
    """
    from methods.adacad import AdaCADDecoding      # type: ignore
    from methods.arr import ARRDecoding            # type: ignore
    from methods.cad import CADDecoding            # type: ignore
    from methods.cocoa import CoCoADecoding        # type: ignore
    from methods.coiecd import COIECDDecoding      # type: ignore

    factories = {
        "cad": lambda: CADDecoding(alpha=kwargs.get("cad_alpha", 0.5)),
        "adacad": lambda: AdaCADDecoding(),
        "cocoa": lambda: CoCoADecoding(
            global_alpha=kwargs.get("cocoa_alpha", config.COCOA_ALPHA),
            gamma=kwargs.get("cocoa_gamma", config.COCOA_GAMMA)),
        "coiecd": lambda: COIECDDecoding(),
        "arr": lambda: ARRDecoding(),
    }
    out = {}
    for name in names:
        if name not in factories:
            raise KeyError(
                f"unknown vendored method {name!r}. Available: "
                f"{sorted(factories)}. Note that 'grd' exists only at HEAD "
                f"({config.VENDOR_HEAD_WITH_GRD}), which does not contain ARR.")
        out[name] = factories[name]()
    return out


# --------------------------------------------------------------------------- #
# Which regime is a method actually in? Measured, not read off its name.
# --------------------------------------------------------------------------- #

def effective_tau(method, logits_ctx, logits_prior) -> float:
    """The tau a method is *really* operating at, recovered from its output.

    Every method in this family is affine in log space:

        adjusted = (1 - tau) * log p_pri + tau * log p_ctx + c

    so tau is recoverable by projecting `adjusted - log p_pri` onto
    `log p_ctx - log p_pri`. Exact for CAD, AdaCAD, CoCoA and the power family;
    for a method that is not affine in log space the residual will be large, which
    is itself the useful answer.

    This exists because a method's self-reported tau cannot be trusted. The repo's
    `CoCoADecoding.get_tau()` returns `global_alpha` (0.5) while the method actually
    operates at `alpha + gamma` (1.5) — the difference between interpolation and
    extrapolation, i.e. the entire distinction the paper is about. Rather than
    trusting either the name or the accessor, measure.

    **Both vectors are mean-centred before projecting**, which fits the intercept
    implicitly. The constant `c` is not incidental: normalising leaves one, and CAD
    and AdaCAD combine *raw* logits rather than log-probabilities, so their output
    differs from the log-space form by the partition functions. Projecting without
    centring would bias tau by `c * sum(basis) / ||basis||^2`, which is not zero over
    a real vocabulary — the estimate would look plausible and be wrong, which is the
    failure mode this whole module is meant to prevent.

    The arithmetic lives in `regime.py` (numpy, no torch) so it is covered by the
    local test suite rather than only by the tests that need a GPU.

    Returns the fitted tau; see `regime_report` for the residual.
    """
    return regime_report(method, logits_ctx, logits_prior)["effective_tau"]


def _to_logprob_arrays(method, logits_ctx, logits_prior):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        adjusted = method.get_next_token_logits(logits_ctx, logits_prior)
        return (adjusted.float().flatten().cpu().numpy(),
                F.log_softmax(logits_ctx.float(), dim=-1).flatten().cpu().numpy(),
                F.log_softmax(logits_prior.float(), dim=-1).flatten().cpu().numpy())


def regime_report(method, logits_ctx, logits_prior) -> dict:
    """Fitted tau, the fit residual, and the regime that implies."""
    from .regime import describe

    adj, lp_ctx, lp_pri = _to_logprob_arrays(method, logits_ctx, logits_prior)

    reported = None
    if hasattr(method, "get_tau"):
        try:
            reported = method.get_tau(logits_ctx, logits_prior)
        except TypeError:
            reported = method.get_tau()
        if reported is not None:
            reported = float(reported)

    out = describe(adj, lp_ctx, lp_pri, reported_tau=reported)
    out["method"] = getattr(method, "name", type(method).__name__)
    return out
