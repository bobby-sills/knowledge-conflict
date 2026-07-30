"""Tests for the Inside-Out knowledge score and the internal predictors.

These run without torch or a GPU. They are the checks that matter most: every
Test 1 and Test 2 number is a transformation of `knowledge_score`, so an error
here is an error in the whole pilot with nothing to reveal it.
"""

import numpy as np
import pytest

from pilot.scoring import (best_layer, divergence_report, first_layer_locked,
                           internal_margin, knowledge_score,
                           knowledge_score_matrix, top1_index, top1_is_correct,
                           trajectory_stability)


class TestKnowledgeScore:
    def test_perfect_ranking(self):
        # gold scores above every distractor -> every pair won
        assert knowledge_score([1.0, 0.5, 0.2], [True, False, False]) == 1.0

    def test_inverted_ranking(self):
        assert knowledge_score([0.0, 0.5, 0.9], [True, False, False]) == 0.0

    def test_middle_ranking(self):
        # gold beats one of two distractors
        assert knowledge_score([0.5, 0.9, 0.1], [True, False, False]) == 0.5

    def test_ties_count_half(self):
        # A tie is not "ranked higher". Counting it as a win would flatter every
        # coarse scorer, and the logit lens produces near-ties at early layers.
        assert knowledge_score([0.5, 0.5], [True, False]) == 0.5
        assert knowledge_score([0.5, 0.5, 0.1], [True, False, False]) == 0.75

    def test_multiple_correct_answers(self):
        # aliases: both correct candidates are compared against both incorrect
        score = knowledge_score([1.0, 0.9, 0.2, 0.1], [True, True, False, False])
        assert score == 1.0

    def test_nan_poisons_the_fact(self):
        # A partially scored candidate set would change the pair denominator
        # between facts, making per-fact scores incomparable.
        assert np.isnan(knowledge_score([1.0, np.nan, 0.2], [True, False, False]))

    def test_no_pairs_is_nan(self):
        assert np.isnan(knowledge_score([1.0, 0.5], [True, True]))
        assert np.isnan(knowledge_score([1.0, 0.5], [False, False]))

    def test_equals_auc(self):
        # The definition is an AUC over the candidate set; check against sklearn.
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = rng.integers(3, 12)
            scores = rng.normal(size=n)
            correct = np.zeros(n, dtype=bool)
            correct[rng.integers(0, n)] = True
            assert knowledge_score(scores, correct) == pytest.approx(
                sk.roc_auc_score(correct, scores))

    def test_matrix_form_matches_row_form(self):
        rng = np.random.default_rng(1)
        mat = rng.normal(size=(5, 6))
        correct = [True, False, False, False, False, False]
        got = knowledge_score_matrix(mat, correct)
        want = [knowledge_score(row, correct) for row in mat]
        assert np.allclose(got, want)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            knowledge_score([1.0, 2.0], [True])


class TestTop1:
    def test_top1_index(self):
        assert top1_index([0.1, 0.9, 0.3]) == 1

    def test_tie_at_top_counts_as_wrong(self):
        # Gold is always candidate 0, so argmax's first-index tie-break would
        # silently credit every tie to gold.
        assert top1_is_correct([0.9, 0.9], [True, False]) is False

    def test_tie_among_correct_is_fine(self):
        assert top1_is_correct([0.9, 0.9, 0.1], [True, True, False]) is True

    def test_clear_win(self):
        assert top1_is_correct([0.9, 0.1], [True, False]) is True
        assert top1_is_correct([0.1, 0.9], [True, False]) is False

    def test_nan_is_wrong(self):
        assert top1_is_correct([np.nan, 0.1], [True, False]) is False


class TestLayerSelection:
    def test_picks_argmax_of_means(self):
        scores = np.array([[0.5, 0.9, 0.6], [0.4, 0.8, 0.7]])
        layer, means = best_layer(scores)
        assert layer == 1
        assert np.allclose(means, [0.45, 0.85, 0.65])

    def test_ignores_nan_rows_per_layer(self):
        scores = np.array([[0.5, np.nan], [0.4, 1.0]])
        layer, means = best_layer(scores)
        assert layer == 1
        assert means[1] == pytest.approx(1.0)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            best_layer(np.full((3, 4), np.nan))


class TestInternalPredictors:
    def test_margin(self):
        assert internal_margin([0.9, 0.4, 0.1]) == pytest.approx(0.5)

    def test_margin_needs_two_candidates(self):
        assert np.isnan(internal_margin([0.9]))

    def test_trajectory_stability_all_layers(self):
        # candidate 0 is top-1 at every layer
        arr = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert trajectory_stability(arr) == 1.0

    def test_trajectory_stability_late_answer(self):
        # final layer picks candidate 1, which was top-1 only at the last layer
        arr = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert trajectory_stability(arr) == pytest.approx(1 / 3)

    def test_trajectory_stability_explicit_index(self):
        arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert trajectory_stability(arr, answer_index=0) == 0.5

    def test_first_layer_locked(self):
        # candidate 1 becomes top-1 at layer 2 and stays
        arr = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        assert first_layer_locked(arr) == 2

    def test_first_layer_locked_never_wavers(self):
        arr = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert first_layer_locked(arr) == 0


class TestDivergenceReport:
    def _record(self, internal, external, state="correction"):
        return {"correct_mask": [True, False], "internal_scores": internal,
                "external_scores": external, "state": state}

    def test_counts_divergence(self):
        records = [
            self._record([1.0, 0.0], [1.0, 0.0]),      # agree
            self._record([1.0, 0.0], [0.0, 1.0]),      # diverge, internal right
            self._record([0.0, 1.0], [1.0, 0.0]),      # diverge, external right
        ]
        rep = divergence_report(records)
        assert rep["n_facts"] == 3
        assert rep["n_diverged"] == 2
        assert rep["divergence_rate"] == pytest.approx(2 / 3)
        assert rep["on_divergent_internal_correct"] == 0.5
        assert rep["on_divergent_external_correct"] == 0.5

    def test_knowledge_gap(self):
        records = [self._record([1.0, 0.0], [0.0, 1.0])]
        rep = divergence_report(records)
        assert rep["internal_knowledge_mean"] == 1.0
        assert rep["external_knowledge_mean"] == 0.0
        assert rep["knowledge_gap"] == 1.0

    def test_skips_unscored_records(self):
        records = [self._record([1.0, 0.0], [1.0, 0.0]),
                   self._record([np.nan, 0.0], [1.0, 0.0])]
        assert divergence_report(records)["n_facts"] == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            divergence_report([])


class TestKillCriterion:
    def test_fires_on_low_divergence(self):
        from pilot.config import KILL
        from pilot.scoring import check_test1_kill
        rep = {"divergence_rate": 0.05, "on_divergent_internal_correct": 0.9,
               "on_divergent_external_correct": 0.1}
        out = check_test1_kill(rep, KILL)
        assert out["fired"] is True
        assert "divergence rate" in out["reasons"][0]

    def test_fires_when_internal_does_not_win(self):
        from pilot.config import KILL
        from pilot.scoring import check_test1_kill
        rep = {"divergence_rate": 0.5, "on_divergent_internal_correct": 0.3,
               "on_divergent_external_correct": 0.4}
        assert check_test1_kill(rep, KILL)["fired"] is True

    def test_passes_when_both_conditions_met(self):
        from pilot.config import KILL
        from pilot.scoring import check_test1_kill
        rep = {"divergence_rate": 0.25, "on_divergent_internal_correct": 0.6,
               "on_divergent_external_correct": 0.2}
        out = check_test1_kill(rep, KILL)
        assert out["fired"] is False
        assert out["reasons"] == []
