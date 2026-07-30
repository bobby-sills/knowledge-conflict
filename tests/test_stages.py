"""The analysis stages, run end to end on synthetic capture files.

These are the tests that catch wiring rather than arithmetic: whether a stage
writes the artefact the next stage reads, whether resuming skips completed work,
whether a kill criterion actually stops `cli all`, and whether the report-split
lock holds. All of it runs from fabricated JSONL, so no model is needed — which is
the same property that lets the real Tests 1, 2, 4 and 3a re-run after a dead
Colab session.
"""

import json

import numpy as np
import pytest

from pilot import config, io_utils


N_LAYERS = 6
GOOD_LAYER = 4


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point the package at a temp project dir and fabricate stage 00-02 output."""
    monkeypatch.setattr(config, "PATHS", config.Paths(tmp_path).mkdirs())
    paths = config.PATHS

    facts, prior_rows, ctx_rows, screen_rows, cases = [], [], [], [], []
    rng = np.random.default_rng(0)

    # 60 facts: half correction, half resistance+agreement. The planted structure
    # is that the internal score at GOOD_LAYER separates the two states and the
    # external score does not, so Test 1 and Test 2 should both come back positive.
    for i in range(60):
        resist = i % 2 == 0
        split = "train" if i % 3 else "layer"
        fid = f"f{i}"
        facts.append({
            "fact_id": fid, "source": "popqa", "subject": f"Subject {i}",
            "subject_key": f"e{i}", "relation": "occupation",
            "object": "mathematician", "question": f"What is Subject {i}'s occupation?",
            "gold_aliases": ["mathematician"],
            "distractors": ["politician", "poet"], "distractor_pop": [10, 20],
            "s_pop": 100 + i, "o_pop": 50, "log_s_pop": 2.0 + 0.01 * i,
            "s_wiki_title": f"Subject {i}", "split": split,
        })

        # Internal: gold on top at GOOD_LAYER for resistance cases only.
        internal = []
        for layer in range(N_LAYERS):
            if layer == GOOD_LAYER and resist:
                internal.append([-0.1, -2.0, -3.0])
            elif layer == GOOD_LAYER:
                internal.append([-2.5, -0.2, -3.0])
            else:
                internal.append(list(rng.normal(-2, 0.3, 3)))
        prior_rows.append({
            "fact_id": fid, "split": split, "correct_mask": [True, False, False],
            "internal_logp_by_layer": internal,
            "external_first_token_logp": [-1.5, -1.4, -2.0],   # uninformative
            "external_full_span": [{"mean_logp": -1.0, "sum_logp": -2.0,
                                    "n_tokens": 2}] * 3,
            "candidate_token_ids": [10, 11, 12],
            "usable_first_token": True, "prior_entropy": 1.0, "prior_max": 0.7,
            "log_popularity": 2.0 + 0.01 * i, "n_layers_captured": N_LAYERS,
            "prior_topk": {"ids": [10, 11], "logprobs": [-1.5, -1.4]},
            "resid_row": i,
        })

        samples = ["mathematician"] * (7 if resist else 0) + \
                  ["politician"] * (1 if resist else 8)
        screen_rows.append({
            "fact_id": fid, "split": split, "samples": samples,
            "hits": [s == "mathematician" for s in samples],
            "n_hits": sum(s == "mathematician" for s in samples), "n_samples": 8,
            "prior_label": "correct" if resist else "wrong",
        })

        states = [("corrupted", "resistance"), ("faithful", "agreement")] if resist \
            else [("faithful", "correction")]
        for variant, state in states:
            cid = f"{fid}::{variant}"
            cases.append({
                "case_id": cid, "fact_id": fid, "split": split, "state": state,
                "doc_variant": variant, "document": f"A passage about Subject {i}.",
                "stated_object": "politician" if variant == "corrupted"
                                 else "mathematician",
                "prior_label": "correct" if resist else "wrong",
            })
            ctx_rows.append({
                "case_id": cid, "fact_id": fid, "split": split, "state": state,
                "doc_variant": variant,
                "stated_object": "politician" if variant == "corrupted"
                                 else "mathematician",
                "gold_token": 10, "stated_token": 11,
                "prior_entropy": 1.0, "ctx_entropy": 0.8, "prior_max": 0.7,
                "ctx_max": 0.9, "entropy_gap": 0.2, "jsd": 0.3, "renyi": 0.6,
                "kl_ctx_pri": 0.5, "log_popularity": 2.0 + 0.01 * i,
                "ctx_argmax": 11, "pri_argmax": 10,
                "ctx_topk": {"ids": [11], "logprobs": [-0.1]},
                "pri_topk": {"ids": [10], "logprobs": [-0.2]},
                "gold_token_": 10,
                "competitor_ctxtop": 11,
                "l_pri_ctxtop": 2.0 if state == "resistance" else -2.0,
                "l_ctx_ctxtop": -2.0 if state == "resistance" else 3.0,
                "tau_star_ctxtop": 0.5 if state == "resistance" else 0.4,
            })

    io_utils.append_jsonl(paths.factset, facts)
    io_utils.append_jsonl(paths.prior, screen_rows)
    io_utils.append_jsonl(paths.states, cases)
    io_utils.append_jsonl(paths.results / "capture_prior.jsonl", prior_rows)
    io_utils.append_jsonl(paths.results / "capture_ctx.jsonl", ctx_rows)
    return paths


class TestReportSplitLock:
    def test_stage03_refuses_the_report_split(self, project):
        from pilot.stages import stage03_test1
        with pytest.raises(io_utils.ReportSplitLocked):
            stage03_test1.run(splits_for_report=("report",))

    def test_stage04_refuses_the_report_split(self, project):
        from pilot.stages import stage04_test2
        with pytest.raises(io_utils.ReportSplitLocked):
            stage04_test2.run(report_split=("report",), layer=GOOD_LAYER)

    def test_stage05_refuses_the_report_split(self, project):
        from pilot.stages import stage05_test3a
        with pytest.raises(io_utils.ReportSplitLocked):
            stage05_test3a.run(splits=("report",))

    def test_lock_fires_before_missing_data(self, project, tmp_path):
        # An illegal split request must not be masked by a mundane failure.
        (tmp_path / "results" / "capture_prior.jsonl").unlink()
        from pilot.stages import stage03_test1
        with pytest.raises(io_utils.ReportSplitLocked):
            stage03_test1.run(splits_for_report=("report",))

    def test_unlocking_lets_it_through(self, project):
        # Past the lock, the stage fails on the ordinary grounds that this fixture
        # has no report-split facts — and says so clearly rather than raising from
        # deep inside the scoring code.
        io_utils.unlock_report(project.report_lock, "tests 0-4 written up")
        from pilot.stages import stage03_test1
        with pytest.raises(SystemExit, match="no captured facts in splits"):
            stage03_test1.run(splits_for_report=("report",),
                              splits_for_layer=("layer",))

    def test_unlocked_report_split_actually_runs(self, project):
        # Add a report-split fact so the unlocked path is exercised for real.
        io_utils.unlock_report(project.report_lock, "tests 0-4 written up")
        row = io_utils.read_jsonl(project.results / "capture_prior.jsonl")[0]
        io_utils.append_jsonl(project.results / "capture_prior.jsonl",
                              [dict(row, fact_id="r1", split="report"),
                               dict(row, fact_id="r2", split="report")])
        from pilot.stages import stage03_test1
        out = stage03_test1.run(splits_for_report=("report",),
                                splits_for_layer=("layer",))
        assert out["n_facts"] == 2


class TestStage03:
    def test_finds_the_planted_layer(self, project):
        from pilot.stages import stage03_test1
        out = stage03_test1.run(splits_for_report=("train",),
                                splits_for_layer=("layer",))
        assert out["chosen_layer"] == GOOD_LAYER

    def test_writes_its_artefact_and_figures(self, project):
        from pilot.stages import stage03_test1
        stage03_test1.run()
        data = io_utils.read_json(project.results / "test1" / "test1.json")
        assert data["chosen_layer"] == GOOD_LAYER
        assert "divergence_rate" in data
        assert (project.figures / "test1_knowledge_by_layer.png").exists()
        assert (project.figures / "test1_scatter.png").exists()

    def test_records_which_split_did_what(self, project):
        from pilot.stages import stage03_test1
        stage03_test1.run()
        data = io_utils.read_json(project.results / "test1" / "test1.json")
        assert data["layer_selection_split"] == ["layer"]
        assert data["report_split_used"] == ["train"]

    def test_logs_a_manifest_entry(self, project):
        from pilot.stages import stage03_test1
        stage03_test1.run()
        stages = [r["stage"] for r in io_utils.read_jsonl(project.manifest)]
        assert "03_test1" in stages

    def test_manifest_records_config_and_kill_thresholds(self, project):
        from pilot.stages import stage03_test1
        stage03_test1.run()
        rec = io_utils.read_jsonl(project.manifest)[-1]
        assert rec["config"]["kill"]["min_auc_margin"] == 0.05
        assert rec["config"]["model_id"] == config.MODEL_ID


class TestStage04:
    def test_produces_an_auc_table_over_conflict_cases_only(self, project):
        from pilot.stages import stage04_test2
        out = stage04_test2.run(fit_split=("train",), report_split=("layer",),
                                layer=GOOD_LAYER)
        data = io_utils.read_json(project.results / "test2" / "test2.json")
        assert data["n_cases"] == data["n_resistance"] + data["n_correction"]
        assert data["n_resistance"] > 0 and data["n_correction"] > 0
        assert out["layer"] == GOOD_LAYER

    def test_internal_signal_wins_on_the_planted_data(self, project):
        # The fixture plants exactly the hypothesis: internal separates the states
        # at GOOD_LAYER, external does not. If the wiring is right, the gate passes.
        from pilot.stages import stage04_test2
        stage04_test2.run(layer=GOOD_LAYER)
        data = io_utils.read_json(project.results / "test2" / "test2.json")
        assert data["auc"]["internal_knowledge"]["auc"] == pytest.approx(1.0)
        assert data["kill"]["best_internal"] == "internal_knowledge"

    def test_every_spec_signal_appears_in_the_table(self, project):
        from pilot.signals import ALL_SIGNALS
        from pilot.stages import stage04_test2
        stage04_test2.run(layer=GOOD_LAYER)
        data = io_utils.read_json(project.results / "test2" / "test2.json")
        assert set(data["auc"]) == set(ALL_SIGNALS)

    def test_permutation_null_is_reported_for_every_signal(self, project):
        from pilot.signals import ALL_SIGNALS
        from pilot.stages import stage04_test2
        stage04_test2.run(layer=GOOD_LAYER)
        data = io_utils.read_json(project.results / "test2" / "test2.json")
        assert set(data["permutation_auc"]) == set(ALL_SIGNALS)

    def test_error_correlation_and_figures_written(self, project):
        from pilot.stages import stage04_test2
        stage04_test2.run(layer=GOOD_LAYER)
        data = io_utils.read_json(project.results / "test2" / "test2.json")
        assert "matrix" in data["error_correlation"]
        assert (project.figures / "test2_auc.png").exists()
        assert (project.figures / "test2_error_correlation.png").exists()
        assert (project.results / "test2" / "bootstrap_draws.npy").exists()

    def test_takes_the_layer_from_stage03_when_not_given(self, project):
        from pilot.stages import stage03_test1, stage04_test2
        stage03_test1.run()
        out = stage04_test2.run()
        assert out["layer"] == GOOD_LAYER

    def test_fails_clearly_without_a_layer(self, project):
        from pilot.stages import stage04_test2
        with pytest.raises(SystemExit, match="stage 03"):
            stage04_test2.run()


class TestStage05:
    def test_summarises_tau_star_by_state(self, project):
        from pilot.stages import stage05_test3a
        stage05_test3a.run(splits=("train", "layer"))
        data = io_utils.read_json(project.results / "test3a" / "test3a.json")
        by_state = data["competitor_is_ctx_top"]
        assert set(by_state) == {"correction", "resistance", "agreement"}
        # The fixture plants tau* in (0,1) for every case.
        assert by_state["resistance"]["frac_in_interpolation"] == 1.0
        assert (project.figures / "test3a_tau_star.png").exists()


class TestCliStatus:
    def test_counts_artefacts(self, project, capsys):
        from pilot.cli import status
        out = status()
        assert out["artefacts"]["factset"] == 60
        assert out["artefacts"]["capture (ctx)"] == 90    # 30 x2 + 30 x1
        assert out["report_unlocked"] is False

    def test_reports_the_unlock(self, project):
        from pilot.cli import status
        io_utils.unlock_report(project.report_lock, "because")
        assert status()["report_unlocked"] is True


class TestKillGating:
    def test_cli_all_stops_at_a_fired_criterion(self, project, monkeypatch):
        from pilot import cli
        # Make stage 03 report a fired criterion, then assert nothing after it ran.
        ran = []

        def fake_stage(name, argv):
            ran.append(name)
            if name == "test1":
                io_utils.write_json(
                    config.PATHS.results / "test1" / "test1.json",
                    {"kill": {"fired": True, "reasons": ["planted"]}})
            return {}

        monkeypatch.setattr(cli, "_run_stage", fake_stage)
        out = cli.run_all()
        assert out["_stopped_at"] == "test1"
        assert ran == ["factset", "prior", "capture", "test1"]
        assert "test2" not in ran

    def test_cli_all_continues_when_nothing_fires(self, project, monkeypatch):
        from pilot import cli
        ran = []
        monkeypatch.setattr(cli, "_run_stage",
                            lambda name, argv: (ran.append(name), {})[1])
        cli.run_all()
        assert ran == cli.PIPELINE


class TestResume:
    def test_stages_skip_completed_work(self, project):
        # The resume contract: done keys come from disk, so a rerun does nothing.
        path = project.results / "capture_prior.jsonl"
        before = len(io_utils.read_jsonl(path))
        done = io_utils.load_done_keys(path, "fact_id")
        facts = io_utils.read_jsonl(project.factset)
        todo = [f for f in facts if f["fact_id"] not in done]
        assert todo == []
        assert len(io_utils.read_jsonl(path)) == before

    def test_a_truncated_capture_file_loses_only_the_last_row(self, project):
        path = project.results / "capture_prior.jsonl"
        rows = io_utils.read_jsonl(path)
        with open(path, "a") as fh:
            fh.write(json.dumps({"fact_id": "partial"})[:20])
        assert len(io_utils.read_jsonl(path)) == len(rows)
