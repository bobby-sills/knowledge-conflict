"""Tests for split hygiene, document construction, and state assignment.

Split-by-entity is the one thing the spec says is painful to retrofit and
invalidates results if wrong, so it gets checked hard: determinism, disjointness,
and stability under a growing fact set.
"""

import pytest

from pilot.documents import (RELATION_TEMPLATES, build_conflict_cases,
                             make_document)
from pilot.factset import candidates, correct_mask, pop_bin
from pilot.splits import assign_split, check_disjoint, entity_bucket, is_dev


def _fact(fid="f1", subject="Ada Lovelace", subject_key="e1",
          relation="occupation", obj="mathematician",
          distractors=("politician", "poet"), split="train", **kw):
    f = {"fact_id": fid, "subject": subject, "subject_key": subject_key,
         "relation": relation, "object": obj, "gold_aliases": [obj],
         "distractors": list(distractors), "split": split,
         "question": f"What is {subject}'s {relation}?",
         "s_wiki_title": subject}
    f.update(kw)
    return f


class TestSplits:
    def test_deterministic(self):
        assert assign_split("e1") == assign_split("e1")
        assert entity_bucket("e1") == entity_bucket("e1")

    def test_bucket_in_unit_interval(self):
        for i in range(200):
            assert 0.0 <= entity_bucket(f"e{i}") < 1.0

    def test_all_three_splits_are_reachable(self):
        got = {assign_split(f"entity-{i}") for i in range(400)}
        assert got == {"train", "layer", "report"}

    def test_proportions_are_roughly_right(self):
        n = 6000
        counts = {"train": 0, "layer": 0, "report": 0}
        for i in range(n):
            counts[assign_split(f"e{i}")] += 1
        assert counts["train"] / n == pytest.approx(0.40, abs=0.03)
        assert counts["layer"] / n == pytest.approx(0.30, abs=0.03)
        assert counts["report"] / n == pytest.approx(0.30, abs=0.03)

    def test_adding_facts_never_moves_an_existing_entity(self):
        # A rebuild with more facts must not reshuffle assignments, or every
        # earlier result silently refers to a different split.
        before = {f"e{i}": assign_split(f"e{i}") for i in range(50)}
        after = {f"e{i}": assign_split(f"e{i}") for i in range(500)}
        for k, v in before.items():
            assert after[k] == v

    def test_bad_fractions_raise(self):
        with pytest.raises(ValueError):
            assign_split("e1", {"train": 0.5, "layer": 0.2, "report": 0.2})

    def test_disjointness_check_passes_on_clean_data(self):
        rows = [{"subject_key": "a", "split": "train"},
                {"subject_key": "a", "split": "train"},
                {"subject_key": "b", "split": "layer"}]
        info = check_disjoint(rows)
        assert info["n_entities"] == 2
        assert info["facts_per_split"] == {"train": 2, "layer": 1}

    def test_disjointness_check_catches_a_leak(self):
        rows = [{"subject_key": "a", "split": "train"},
                {"subject_key": "a", "split": "report"}]
        with pytest.raises(AssertionError, match="leaks across splits"):
            check_disjoint(rows)

    def test_dev_excludes_report(self):
        assert is_dev("train") and is_dev("layer")
        assert not is_dev("report")


class TestDocuments:
    def test_faithful_states_the_gold(self):
        doc, stated = make_document(_fact(), "faithful")
        assert stated == "mathematician"
        assert "mathematician" in doc

    def test_corrupted_states_a_distractor(self):
        doc, stated = make_document(_fact(), "corrupted")
        assert stated == "politician"
        assert "politician" in doc
        assert "mathematician" not in doc

    def test_deterministic_across_reruns(self):
        # A resumed session must regenerate byte-identical documents.
        a, _ = make_document(_fact(), "corrupted")
        b, _ = make_document(_fact(), "corrupted")
        assert a == b

    def test_variants_have_similar_length(self):
        # If the corrupted passage were less fluent or much shorter, every
        # resistance result would be confounded by that rather than by conflict.
        faithful, _ = make_document(_fact(), "faithful")
        corrupted, _ = make_document(_fact(), "corrupted")
        assert abs(len(faithful) - len(corrupted)) < 30

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            make_document(_fact(), "sideways")

    def test_corrupting_without_a_distractor_raises(self):
        with pytest.raises(ValueError):
            make_document(_fact(distractors=()), "corrupted")

    def test_fallback_template_for_unknown_relation(self):
        doc, stated = make_document(_fact(relation="favourite colour"), "faithful")
        assert "mathematician" in doc
        assert "favourite colour" in doc

    def test_all_popqa_relations_have_a_template(self):
        # PopQA's 16 relations. A missing one silently falls back to the generic
        # template, which reads less like an encyclopedia and weakens the document.
        expected = {"occupation", "place of birth", "genre", "father", "country",
                    "producer", "director", "capital of", "screenwriter",
                    "composer", "color", "religion", "sport", "author", "mother",
                    "capital"}
        assert expected <= set(RELATION_TEMPLATES)


class TestConflictCases:
    def test_wrong_prior_yields_correction_only(self):
        cases = build_conflict_cases([_fact()], {"f1": "wrong"})
        assert [c["state"] for c in cases] == ["correction"]
        assert cases[0]["doc_variant"] == "faithful"

    def test_correct_prior_yields_resistance_and_agreement(self):
        cases = build_conflict_cases([_fact()], {"f1": "correct"})
        assert {c["state"] for c in cases} == {"resistance", "agreement"}
        by_state = {c["state"]: c for c in cases}
        assert by_state["resistance"]["doc_variant"] == "corrupted"
        assert by_state["agreement"]["doc_variant"] == "faithful"

    def test_ambiguous_prior_yields_nothing(self):
        assert build_conflict_cases([_fact()], {"f1": "ambiguous"}) == []

    def test_unscreened_fact_yields_nothing(self):
        assert build_conflict_cases([_fact()], {}) == []

    def test_case_ids_are_unique(self):
        facts = [_fact("f1", subject_key="e1"), _fact("f2", subject_key="e2")]
        cases = build_conflict_cases(facts, {"f1": "correct", "f2": "correct"})
        assert len({c["case_id"] for c in cases}) == len(cases) == 4

    def test_split_is_carried_through(self):
        cases = build_conflict_cases([_fact(split="layer")], {"f1": "correct"})
        assert all(c["split"] == "layer" for c in cases)


class TestPriorLabelling:
    def test_thresholds(self):
        from pilot.prior import label_prior
        assert label_prior(8) == "correct"
        assert label_prior(6) == "correct"
        assert label_prior(5) == "ambiguous"
        assert label_prior(1) == "ambiguous"
        assert label_prior(0) == "wrong"

    def test_fact_seed_is_stable_across_processes(self):
        # `hash()` on a str is salted per interpreter, so using it here would give
        # a fact different samples every session — silently breaking the
        # reproducibility the seed exists to provide.
        import subprocess
        import sys
        from pilot.prior import fact_seed
        here = fact_seed("popqa-12345")
        code = ("import sys; sys.path.insert(0, '.');"
                "from pilot.prior import fact_seed; print(fact_seed('popqa-12345'))")
        env_runs = set()
        for salt in ("0", "1", "random"):
            out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                 text=True, env={"PYTHONHASHSEED": salt,
                                                 "PATH": "/usr/bin:/bin"})
            env_runs.add(out.stdout.strip())
        assert env_runs == {str(here)}, env_runs

    def test_fact_seed_differs_between_facts(self):
        from pilot.prior import fact_seed
        seeds = {fact_seed(f"popqa-{i}") for i in range(200)}
        assert len(seeds) == 200


class TestCandidates:
    def test_gold_is_always_index_zero(self):
        assert candidates(_fact())[0] == "mathematician"

    def test_correct_mask_marks_only_the_gold(self):
        assert correct_mask(_fact()) == [True, False, False]

    def test_aliases_count_as_correct(self):
        f = _fact(distractors=("politician", "mathematician "))
        # Trailing whitespace and case must not create a false distractor.
        assert correct_mask(f) == [True, False, True]

    def test_pop_bins(self):
        assert pop_bin(50) == "[0,100)"
        assert pop_bin(5000) == "[1000,10000)"
        assert pop_bin(10 ** 9) == "[100000,inf)"
