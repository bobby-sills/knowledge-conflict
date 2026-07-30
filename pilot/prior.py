"""Test 0: prior calibration and conflict-state labelling.

Ask the model closed-book N=8 times at temperature 0.7 and count how often the
gold object appears:

    >= 6/8  -> prior correct
    == 0/8  -> prior wrong
    else    -> ambiguous (dropped from analysis, kept in the counts)

The gap between 6/8 and 0/8 is doing real work. A fact the model produces 3/8
times is neither a belief it holds nor one it lacks, and calling it either would
put noise on both sides of Test 2's binary. The cost is throwing away the middle
of the distribution, so the ambiguous fraction is reported: if it is most of the
set, the thresholds are wrong for this model rather than the model being
uncertain, and the spec explicitly asks us to flag that rather than decide it
silently.

Matching is substring-after-normalisation against the gold object and its
aliases, using the vendored normaliser. Strict EM would score "He was a
politician." as a miss on a fact the model plainly knows; the looser rule risks
crediting an answer that merely mentions the gold in passing, which is the right
error to prefer when the label is "does the model hold this belief".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from . import config
from .factset import is_hit
from .documents import prior_prompt

# torch is imported lazily inside `screen_fact`. The labelling rules, the Test 0
# summary and the TriState cross-check are pure functions over saved samples, and
# they stay importable on a machine with no torch — the summary is re-derivable
# from `prior_samples.jsonl` without a GPU, so requiring one would be gratuitous.
if TYPE_CHECKING:
    from .model import ModelBundle


def label_prior(n_hits: int, n_samples: int = config.PRIOR_N_SAMPLES,
                correct_min: int = config.PRIOR_CORRECT_MIN,
                wrong_max: int = config.PRIOR_WRONG_MAX) -> str:
    if n_hits >= correct_min:
        return "correct"
    if n_hits <= wrong_max:
        return "wrong"
    return "ambiguous"


def fact_seed(fact_id: str, seed: int = config.PRIOR_SEED) -> int:
    """Per-fact sampling seed, stable across processes and sessions.

    `hash()` on a string is salted per interpreter (PYTHONHASHSEED), so using it
    here would give a fact different samples in every session — silently breaking
    the reproducibility this function exists to provide. blake2b is stable.
    """
    import hashlib
    digest = hashlib.blake2b(fact_id.encode("utf-8"), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "big")) % (2 ** 31)


def screen_fact(bundle: "ModelBundle", fact: dict,
                n_samples: int = config.PRIOR_N_SAMPLES,
                seed: int = config.PRIOR_SEED) -> dict:
    """Closed-book screening for one fact.

    The seed is derived from the fact id so a resumed session reproduces the same
    samples for the same fact. A global counter would make the samples depend on
    how many facts happened to be processed before the session died.
    """
    from .model import sample_answers

    prompt = prior_prompt(fact, bundle.tokenizer)
    seed_for_fact = fact_seed(fact["fact_id"], seed)
    samples = sample_answers(bundle, prompt, n=n_samples, seed=seed_for_fact)
    hits = [is_hit(s, fact) for s in samples]
    n_hits = sum(hits)
    return {
        "fact_id": fact["fact_id"],
        "split": fact["split"],
        "prompt": prompt,
        "samples": samples,
        "hits": hits,
        "n_hits": n_hits,
        "n_samples": n_samples,
        "prior_label": label_prior(n_hits, n_samples),
        "seed": seed_for_fact,
    }


def summarise_screening(rows: Sequence[dict], facts_by_id: dict[str, dict]) -> dict:
    """Test 0's report: state counts, ambiguous fraction, popularity per state.

    The popularity distribution per state is not decoration. If resistance cases
    are all high-popularity entities and correction cases all low-popularity ones,
    then log-popularity alone separates the Test 2 binary, and any internal signal
    that correlates with popularity will look predictive without carrying
    information about the model's internal state. Better to know that here than to
    discover it in the Test 2 control.
    """
    from .factset import pop_bin

    labels: dict[str, int] = {}
    states: dict[str, int] = {}
    pop_by_state: dict[str, dict[str, int]] = {}
    log_pop_by_state: dict[str, list[float]] = {}
    n_no_distractor = 0

    for r in rows:
        label = r["prior_label"]
        labels[label] = labels.get(label, 0) + 1
        fact = facts_by_id.get(r["fact_id"])
        if fact is None:
            continue
        if label == "wrong":
            case_states = ["correction"]
        elif label == "correct":
            case_states = ["resistance", "agreement"]
            if not fact.get("distractors"):
                case_states = ["agreement"]
                n_no_distractor += 1
        else:
            case_states = []
        for state in case_states:
            states[state] = states.get(state, 0) + 1
            if fact.get("s_pop") is not None:
                b = pop_bin(fact["s_pop"])
                pop_by_state.setdefault(state, {})
                pop_by_state[state][b] = pop_by_state[state].get(b, 0) + 1
                log_pop_by_state.setdefault(state, []).append(fact["log_s_pop"])

    n = sum(labels.values()) or 1
    summary = {
        "n_facts_screened": sum(labels.values()),
        "prior_labels": labels,
        "ambiguous_fraction": labels.get("ambiguous", 0) / n,
        "state_counts": states,
        "popularity_bins_by_state": pop_by_state,
        "n_correct_without_distractor": n_no_distractor,
    }
    if log_pop_by_state:
        import statistics
        summary["median_log_s_pop_by_state"] = {
            state: round(statistics.median(v), 4)
            for state, v in sorted(log_pop_by_state.items())}
    return summary


def agreement_with_tristate(our_rows: Sequence[dict],
                            tristate_facts: Sequence[dict]) -> dict:
    """Cross-check our 6/8-and-0/8 labelling against TriState-Bench's GAPS labels.

    TriState-Bench ships prior-screened states for our exact model, produced by a
    different procedure (Greedy-Anchored Prior Screening). Where the two disagree,
    at least one is wrong, and the disagreement rate is the closest thing to an
    external validity check on the thresholds that we can get without more
    annotation. Only meaningful when the same facts appear in both, which is why
    this returns `n_shared` and refuses to interpret an empty overlap.
    """
    ours = {r["fact_id"]: r["prior_label"] for r in our_rows}
    theirs = {}
    for f in tristate_facts:
        expected = "correct" if f["state"] in ("resistance", "agreement") else "wrong"
        theirs[f["fact_id"]] = expected
    shared = set(ours) & set(theirs)
    if not shared:
        return {"n_shared": 0,
                "note": ("no shared fact ids: TriState-Bench facts are not PopQA "
                         "facts, so this check needs our screening to be run over "
                         "the TriState fact set as well (stage 01 --factset tristate)")}
    agree = sum(1 for k in shared if ours[k] == theirs[k])
    return {"n_shared": len(shared), "agreement": agree / len(shared),
            "n_disagree": len(shared) - agree}
