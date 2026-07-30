"""Tests for the crash-tolerance the whole pipeline depends on.

Colab sessions die mid-write. These tests simulate that: truncated final lines,
duplicate rows from a batch that landed twice, a residual memmap with a short
write. If any of this is wrong, a two-week pilot becomes a two-week pilot that
has to start over.
"""

import json

import numpy as np
import pytest

from pilot.io_utils import (ReportSplitLocked, ResidualStore, append_jsonl,
                            assert_report_unlocked, batched, dedup_jsonl,
                            load_done_keys, read_json, read_jsonl,
                            rewrite_jsonl, unlock_report, write_json)


class TestJsonl:
    def test_append_and_read(self, tmp_path):
        p = tmp_path / "a.jsonl"
        assert append_jsonl(p, [{"fact_id": "x"}, {"fact_id": "y"}]) == 2
        assert append_jsonl(p, [{"fact_id": "z"}]) == 1
        assert [r["fact_id"] for r in read_jsonl(p)] == ["x", "y", "z"]

    def test_empty_append_is_a_noop(self, tmp_path):
        p = tmp_path / "a.jsonl"
        assert append_jsonl(p, []) == 0
        assert not p.exists()

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_jsonl(tmp_path / "nope.jsonl") == []

    def test_truncated_final_line_is_dropped(self, tmp_path):
        # Exactly what a session killed mid-write leaves behind.
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x"}])
        with open(p, "a") as fh:
            fh.write('{"fact_id": "part')
        rows = read_jsonl(p)
        assert len(rows) == 1
        assert rows[0]["fact_id"] == "x"

    def test_nan_becomes_null_not_bare_nan(self, tmp_path):
        # Bare NaN is invalid JSON; pandas.read_json and jq both reject it.
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x", "v": float("nan"),
                          "w": float("inf"), "nested": {"a": [float("nan"), 1.0]}}])
        raw = p.read_text()
        assert "NaN" not in raw and "Infinity" not in raw
        row = json.loads(raw.strip())
        assert row["v"] is None and row["w"] is None
        assert row["nested"]["a"] == [None, 1.0]

    def test_numpy_scalars_serialise(self, tmp_path):
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x", "i": np.int64(3),
                          "f": np.float32(0.5), "arr": np.arange(3)}])
        row = read_jsonl(p)[0]
        assert row["i"] == 3 and row["f"] == 0.5 and row["arr"] == [0, 1, 2]

    def test_done_keys_drive_resume(self, tmp_path):
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x"}, {"fact_id": "y"}])
        assert load_done_keys(p, "fact_id") == {"x", "y"}
        todo = [f for f in ["x", "y", "z"] if f not in load_done_keys(p, "fact_id")]
        assert todo == ["z"]

    def test_dedup_keeps_the_last_row(self, tmp_path):
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x", "v": 1}, {"fact_id": "y", "v": 2},
                         {"fact_id": "x", "v": 3}])
        assert dedup_jsonl(p, "fact_id") == 1
        rows = {r["fact_id"]: r["v"] for r in read_jsonl(p)}
        assert rows == {"x": 3, "y": 2}

    def test_dedup_is_a_noop_when_clean(self, tmp_path):
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x"}, {"fact_id": "y"}])
        assert dedup_jsonl(p, "fact_id") == 0

    def test_rewrite_is_atomic(self, tmp_path):
        p = tmp_path / "a.jsonl"
        append_jsonl(p, [{"fact_id": "x"}])
        rewrite_jsonl(p, [{"fact_id": "q"}])
        assert [r["fact_id"] for r in read_jsonl(p)] == ["q"]
        assert not (tmp_path / "a.jsonl.tmp").exists()


class TestJson:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "d.json"
        write_json(p, {"a": 1, "b": [1, 2]})
        assert read_json(p) == {"a": 1, "b": [1, 2]}

    def test_missing_returns_default(self, tmp_path):
        assert read_json(tmp_path / "nope.json", default={"x": 1}) == {"x": 1}


class TestResidualStore:
    def test_append_and_read_rows(self, tmp_path):
        store = ResidualStore(tmp_path / "r.npy", tmp_path / "r.parquet",
                              n_layers=4, hidden=8)
        a = np.random.default_rng(0).normal(size=(3, 4, 8)).astype(np.float16)
        start, end = store.append(a)
        assert (start, end) == (0, 3)
        assert store.n_rows == 3
        assert np.allclose(store.read_rows([0, 2]), a[[0, 2]].astype(np.float32))

    def test_appends_accumulate_across_sessions(self, tmp_path):
        p, ip = tmp_path / "r.npy", tmp_path / "r.parquet"
        s1 = ResidualStore(p, ip, n_layers=2, hidden=3)
        s1.append(np.ones((2, 2, 3), dtype=np.float16))
        # New session: reopen from the metadata on disk.
        s2 = ResidualStore(p, ip)
        assert s2.n_rows == 2
        s2.append(np.zeros((1, 2, 3), dtype=np.float16))
        assert s2.n_rows == 3
        assert np.allclose(s2.read_rows([0]), 1.0)
        assert np.allclose(s2.read_rows([2]), 0.0)

    def test_shape_mismatch_raises(self, tmp_path):
        store = ResidualStore(tmp_path / "r.npy", tmp_path / "r.parquet",
                              n_layers=4, hidden=8)
        with pytest.raises(ValueError):
            store.append(np.zeros((2, 5, 8), dtype=np.float16))

    def test_reopening_with_wrong_dims_raises(self, tmp_path):
        p, ip = tmp_path / "r.npy", tmp_path / "r.parquet"
        ResidualStore(p, ip, n_layers=4, hidden=8)
        with pytest.raises(ValueError, match="layer mismatch"):
            ResidualStore(p, ip, n_layers=6, hidden=8)
        with pytest.raises(ValueError, match="hidden-dim mismatch"):
            ResidualStore(p, ip, n_layers=4, hidden=16)

    def test_truncate_repairs_a_partial_write(self, tmp_path):
        store = ResidualStore(tmp_path / "r.npy", tmp_path / "r.parquet",
                              n_layers=2, hidden=3)
        store.append(np.ones((4, 2, 3), dtype=np.float16))
        store.truncate_to(2)
        assert store.n_rows == 2

    def test_index_roundtrip(self, tmp_path):
        store = ResidualStore(tmp_path / "r.npy", tmp_path / "r.parquet",
                              n_layers=2, hidden=3)
        store.append(np.ones((2, 2, 3), dtype=np.float16))
        store.write_index([{"fact_id": "a", "row": 0}, {"fact_id": "b", "row": 1}])
        df = store.read_index()
        assert list(df["fact_id"]) == ["a", "b"]
        assert list(df["row"]) == [0, 1]

    def test_expected_size_matches_the_spec_estimate(self, tmp_path):
        # The spec's estimate: 32 layers x 4096 dims x 2k facts in fp16 ~= 0.5 GB.
        store = ResidualStore(tmp_path / "r.npy", tmp_path / "r.parquet",
                              n_layers=33, hidden=4096)
        assert store.row_bytes * 2000 / 1e9 == pytest.approx(0.54, abs=0.02)


class TestReportLock:
    def test_locked_by_default(self, tmp_path):
        with pytest.raises(ReportSplitLocked, match="locked"):
            assert_report_unlocked(tmp_path / "lock", "a test")

    def test_unlock_then_pass(self, tmp_path):
        lock = tmp_path / "lock"
        unlock_report(lock, "tests 0-4 written up")
        assert_report_unlocked(lock, "a test")     # must not raise
        assert read_json(lock)["reason"] == "tests 0-4 written up"


class TestBatched:
    def test_splits_evenly(self):
        assert list(batched([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_last_batch_is_short(self):
        assert list(batched([1, 2, 3], 2)) == [[1, 2], [3]]

    def test_empty(self):
        assert list(batched([], 3)) == []
