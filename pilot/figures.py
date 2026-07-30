"""Every plot named in the spec's Report sections.

One function per figure, each taking already-computed numbers and returning the
path it wrote. No analysis happens here — a plotting function that also computes
its numbers is a plotting function whose numbers cannot be checked.

Colour: conflict states get a fixed mapping used in every figure, so a reader who
learns it once in Figure 1 keeps it for the rest of the document. Correction and
resistance are the two states that carry the argument, so they get the two
strongest, most separable colours; agreement is deliberately quiet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")               # headless: Colab and CI both lack a display
import matplotlib.pyplot as plt
import numpy as np

STATE_COLOURS = {
    "correction": "#1f77b4",     # blue
    "resistance": "#d62728",     # red
    "agreement": "#999999",      # grey, on purpose
}
INTERNAL_COLOUR = "#2ca02c"
EXTERNAL_COLOUR = "#7f7f7f"


def _finish(fig, path: Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


# --------------------------------------------------------------------------- #
# Test 1
# --------------------------------------------------------------------------- #

def knowledge_by_layer(per_layer_internal: np.ndarray,
                       external_mean: float,
                       chosen_layer: int | None,
                       path: Path,
                       title: str = "Internal vs external knowledge by layer") -> Path:
    """Mean internal knowledge score per layer, with external as a flat reference.

    External is a horizontal line because it has no layer: it is read off the
    output distribution, which is what makes "internal exceeds external" a claim
    about depth rather than about two competing scorers.
    """
    means = np.nanmean(per_layer_internal, axis=0)
    sems = (np.nanstd(per_layer_internal, axis=0)
            / np.sqrt(np.sum(np.isfinite(per_layer_internal), axis=0)))
    layers = np.arange(len(means))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, means, color=INTERNAL_COLOUR, lw=2, label="internal (logit lens)")
    ax.fill_between(layers, means - sems, means + sems, color=INTERNAL_COLOUR,
                    alpha=0.2, lw=0)
    ax.axhline(external_mean, color=EXTERNAL_COLOUR, ls="--", lw=1.5,
               label=f"external (output probs) = {external_mean:.3f}")
    ax.axhline(0.5, color="black", ls=":", lw=1, label="chance")
    if chosen_layer is not None:
        ax.axvline(chosen_layer, color=INTERNAL_COLOUR, ls="-.", lw=1, alpha=0.7)
        ax.annotate(f"layer {chosen_layer}", (chosen_layer, ax.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points", fontsize=9,
                    color=INTERNAL_COLOUR)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("knowledge score")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def internal_vs_external_scatter(internal: Sequence[float],
                                 external: Sequence[float],
                                 states: Sequence[str],
                                 path: Path,
                                 title: str = "Internal vs external knowledge") -> Path:
    """Candidate Figure 1: per-fact internal against external, coloured by state.

    The diagonal is the whole point — every fact above it is one where the
    internals rank the correct answer better than the output distribution does.
    Jitter is applied because both axes are pair fractions over a 9-candidate set,
    so values collapse onto a small grid and hundreds of facts hide under one
    marker.
    """
    x = np.asarray(external, dtype=np.float64)
    y = np.asarray(internal, dtype=np.float64)
    rng = np.random.default_rng(0)
    jitter = 0.006
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", alpha=0.6, zorder=1)
    for state in ("agreement", "correction", "resistance"):
        m = np.asarray([s == state for s in states], dtype=bool)
        if not m.any():
            continue
        ax.scatter(x[m] + rng.normal(0, jitter, m.sum()),
                   y[m] + rng.normal(0, jitter, m.sum()),
                   s=14, alpha=0.55, lw=0, label=f"{state} (n={int(m.sum())})",
                   color=STATE_COLOURS[state], zorder=2)
    above = float(np.nanmean(y > x)) if x.size else float("nan")
    ax.set_xlabel("external knowledge score")
    ax.set_ylabel("internal knowledge score")
    ax.set_title(f"{title}\n{above:.1%} of facts above the diagonal", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Test 2
# --------------------------------------------------------------------------- #

def auc_table(boot: Mapping, internal_names: Sequence[str], path: Path,
              title: str = "Resist-or-correct AUC") -> Path:
    """Horizontal AUC bars with bootstrap CIs, internal signals highlighted."""
    per = boot["per_signal"]
    names = sorted(per, key=lambda n: (np.nan_to_num(per[n]["auc"], nan=-1)))
    vals = [per[n]["auc"] for n in names]
    los = [per[n]["auc"] - per[n]["lo"] for n in names]
    his = [per[n]["hi"] - per[n]["auc"] for n in names]
    colours = [INTERNAL_COLOUR if n in internal_names else EXTERNAL_COLOUR
               for n in names]

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(names) + 1.6))
    ax.barh(names, vals, color=colours, alpha=0.85,
            xerr=[los, his], error_kw={"lw": 1, "capsize": 3, "ecolor": "black"})
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_xlabel("AUC (predicting 'resist')")
    ax.set_xlim(0.3, 1.0)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    for i, n in enumerate(names):
        ax.text(vals[i] + 0.008, i, f"{vals[i]:.3f}", va="center", fontsize=8)
    return _finish(fig, path)


def error_correlation(matrix_info: Mapping, path: Path,
                      title: str = "Error correlation between predictors") -> Path:
    """The matrix that decides "complementary" versus "simply worse".

    Diverging colormap centred at zero, because the sign is the finding: near-zero
    off-diagonals between the internal signal and the external ones mean the
    internal signal fails on different questions, which keeps an ensemble method
    alive even when Test 2's gate has fired.
    """
    names = matrix_info["names"]
    mat = np.asarray(matrix_info["matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(0.55 * len(names) + 3, 0.55 * len(names) + 2.4))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i, j]
            label = "--" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and abs(v) > 0.6 else "black")
    if matrix_info.get("degenerate"):
        # A constant error vector has no correlation with anything; saying so on
        # the figure stops a blank cell being read as "uncorrelated".
        ax.set_xlabel("'--' = undefined (constant errors): "
                      + ", ".join(matrix_info["degenerate"]), fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="phi correlation of errors")
    ax.set_title(title, fontsize=11)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Test 3
# --------------------------------------------------------------------------- #

def tau_star_histogram(taus_by_state: Mapping[str, Sequence[float]], path: Path,
                       title: str = "Pairwise reversal threshold tau*") -> Path:
    """Test 3a. The shaded band is the interpolation regime, tau in (0, 1).

    Values outside [-0.5, 2.5] are clipped *into the edge bins* and the clipped
    count is printed, rather than silently dropped: the fraction of cases with no
    crossover in the interpolation regime is part of the finding.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    lo, hi = -0.5, 2.5
    bins = np.linspace(lo, hi, 61)
    ax.axvspan(0, 1, color="#ffe08a", alpha=0.35, lw=0,
               label="interpolation, tau in (0,1)")
    n_clipped = 0
    for state in ("correction", "resistance", "agreement"):
        vals = np.asarray([v for v in taus_by_state.get(state, [])
                           if v is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        n_clipped += int(((vals < lo) | (vals > hi)).sum())
        ax.hist(np.clip(vals, lo, hi), bins=bins, histtype="step", lw=1.8,
                color=STATE_COLOURS[state], label=f"{state} (n={vals.size})")
    ax.axvline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("tau*")
    ax.set_ylabel("cases")
    ax.set_title(f"{title}\n{n_clipped} values clipped into the edge bins",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def tau_sweep_curves(sweep: Mapping[float, Mapping[str, float]], path: Path,
                     oracle: Mapping[str, float] | None = None,
                     baselines: Mapping[str, Mapping[str, float]] | None = None,
                     title: str = "Per-state EM across the power family") -> Path:
    """Test 3b's trade-off curves: EM per state as tau sweeps 0 -> 2.5.

    The crossing of the correction and resistance curves is the regime asymmetry
    the whole project is about: no single tau is at the top of both. Baselines are
    horizontal lines because their effective tau varies per token, so they have no
    x-position — they are levels to clear, not points on the curve.
    """
    taus = sorted(sweep)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for state in ("correction", "resistance", "agreement"):
        ys = [sweep[t].get(state, np.nan) for t in taus]
        if all(np.isnan(y) for y in ys):
            continue
        ax.plot(taus, ys, color=STATE_COLOURS[state], lw=2, label=state)
    ys = [sweep[t].get("overall", np.nan) for t in taus]
    ax.plot(taus, ys, color="black", lw=1.5, ls="--", label="overall")

    if baselines:
        for i, (name, scores) in enumerate(sorted(baselines.items())):
            y = scores.get("overall")
            if y is None:
                continue
            ax.axhline(y, color="#555555", lw=0.9, ls=(0, (2, 3)), alpha=0.8)
            ax.annotate(f"{name} {y:.3f}", (taus[-1], y), fontsize=7,
                        xytext=(-2, 2 + 0 * i), textcoords="offset points",
                        ha="right", color="#333333")
    if oracle and oracle.get("overall") is not None:
        ax.axhline(oracle["overall"], color="#8c564b", lw=1.6,
                   label=f"oracle 3-way = {oracle['overall']:.3f}")

    ax.axvspan(0, 1, color="#ffe08a", alpha=0.25, lw=0)
    ax.axvline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("tau  (0 = pure prior, 1 = pure context, >1 = extrapolation)")
    ax.set_ylabel("EM")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def correction_vs_resistance(sweep: Mapping[float, Mapping[str, float]],
                             path: Path,
                             oracle: Mapping[str, float] | None = None,
                             baselines: Mapping[str, Mapping[str, float]] | None = None,
                             title: str = "Correction vs resistance trade-off") -> Path:
    """The same data as a frontier: correction EM on x, resistance EM on y.

    Every fixed tau is one point on a curve that no method can leave without a
    per-question signal. A method above and to the right of that curve is doing
    something a global tau cannot — which is exactly the claim being tested, made
    visual.
    """
    taus = sorted(sweep)
    xs = [sweep[t].get("correction", np.nan) for t in taus]
    ys = [sweep[t].get("resistance", np.nan) for t in taus]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(xs, ys, color="#444444", lw=1.5, marker="o", ms=3, alpha=0.8,
            label="fixed tau frontier")
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        if t in sweep:
            ax.annotate(f"{t:g}", (sweep[t].get("correction", np.nan),
                                   sweep[t].get("resistance", np.nan)),
                        fontsize=7, xytext=(4, -8), textcoords="offset points")
    if baselines:
        for name, s in sorted(baselines.items()):
            if s.get("correction") is None or s.get("resistance") is None:
                continue
            ax.scatter(s["correction"], s["resistance"], marker="s", s=42,
                       color=EXTERNAL_COLOUR, zorder=3)
            ax.annotate(name, (s["correction"], s["resistance"]), fontsize=7,
                        xytext=(5, 4), textcoords="offset points")
    if oracle and oracle.get("correction") is not None:
        ax.scatter(oracle["correction"], oracle["resistance"], marker="*", s=200,
                   color="#8c564b", zorder=4, label="oracle 3-way")
    ax.set_xlabel("correction EM")
    ax.set_ylabel("resistance EM")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Test 4
# --------------------------------------------------------------------------- #

def permutation_control(real: Mapping[float, Mapping[str, float]],
                        permuted: Sequence[Mapping[float, Mapping[str, float]]],
                        path: Path, metric: str = "overall",
                        title: str = "Permutation control") -> Path:
    """Real routed curve against the shuffled-signal curves.

    If the grey band swallows the coloured line, the per-question adaptivity was
    doing nothing and the signal was a global knob. This is the plot that is
    supposed to be boring; when it is not boring, the result is not real.
    """
    taus = sorted(real)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if permuted:
        stack = np.array([[p.get(t, {}).get(metric, np.nan) for t in taus]
                          for p in permuted], dtype=np.float64)
        lo = np.nanpercentile(stack, 5, axis=0)
        hi = np.nanpercentile(stack, 95, axis=0)
        ax.fill_between(taus, lo, hi, color="#999999", alpha=0.35, lw=0,
                        label=f"shuffled signal, 5-95% ({len(permuted)} draws)")
        ax.plot(taus, np.nanmean(stack, axis=0), color="#666666", lw=1.2, ls="--",
                label="shuffled mean")
    ax.plot(taus, [real[t].get(metric, np.nan) for t in taus],
            color=INTERNAL_COLOUR, lw=2.2, label="real signal")
    ax.set_xlabel("tau")
    ax.set_ylabel(f"{metric} EM")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Test 0
# --------------------------------------------------------------------------- #

def state_counts(summary: Mapping, path: Path,
                 title: str = "Conflict states and prior labels") -> Path:
    """Test 0's diagnostic: how many cases of each state, and the ambiguous share."""
    labels = summary.get("prior_labels", {})
    states = summary.get("state_counts", {})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    order_l = [k for k in ("correct", "wrong", "ambiguous") if k in labels]
    axes[0].bar(order_l, [labels[k] for k in order_l],
                color=["#2ca02c", "#d62728", "#cccccc"][:len(order_l)])
    axes[0].set_title(f"prior labels (ambiguous "
                      f"{summary.get('ambiguous_fraction', float('nan')):.1%})",
                      fontsize=10)
    order_s = [k for k in ("correction", "resistance", "agreement") if k in states]
    axes[1].bar(order_s, [states[k] for k in order_s],
                color=[STATE_COLOURS[k] for k in order_s])
    axes[1].set_title("conflict cases per state", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    return _finish(fig, path)


def popularity_by_state(log_pop_by_state: Mapping[str, Sequence[float]], path: Path,
                        title: str = "Entity popularity by conflict state") -> Path:
    """The confound check. Separated distributions here mean log-popularity alone
    predicts the Test 2 binary, and any popularity-correlated signal inherits that
    for free."""
    states = [s for s in ("correction", "resistance", "agreement")
              if log_pop_by_state.get(s)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for state in states:
        vals = np.asarray(log_pop_by_state[state], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=30, histtype="step", lw=1.8, density=True,
                color=STATE_COLOURS[state], label=f"{state} (n={vals.size})")
    ax.set_xlabel("log10 subject pageviews")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)
