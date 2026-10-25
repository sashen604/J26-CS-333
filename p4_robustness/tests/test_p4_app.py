"""Tests for app/p4_app.py helper functions (not the Streamlit rendering). No data or model needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import p4_app  # noqa: E402  (first: puts experiments/ on sys.path)
import data_prep  # noqa: E402
import drift_experiment as de  # noqa: E402
import leakage_audit  # noqa: E402


# --- overlap -----------------------------------------------------------------
def test_overlap_identical_runs_is_one():
    runs = [["a", "b", "c", "d", "e"]] * 4
    assert p4_app.common_top_k_share(runs) == 1.0


def test_overlap_counts_features_common_to_all_runs():
    runs = [["a", "b", "c", "d", "e"], ["a", "b", "c", "x", "y"], ["e", "c", "a", "b", "z"]]
    assert p4_app.common_top_k_share(runs) == 3 / 5  # a, b, c


def test_overlap_ignores_order_and_handles_edge_cases():
    assert p4_app.common_top_k_share([["a", "b", "c", "d", "e"], ["e", "d", "c", "b", "a"]]) == 1.0
    assert p4_app.common_top_k_share([["a", "b", "c", "d", "e"], ["v", "w", "x", "y", "z"]]) == 0.0
    assert p4_app.common_top_k_share([["a", "b", "c", "d", "e"]]) == 1.0
    assert p4_app.common_top_k_share([]) is None


# --- reading h and L ---------------------------------------------------------
def test_committed_h_L_reads_calibrated_rows_only(tmp_path):
    md = tmp_path / "drift_results.md"
    md.write_text("| Detector | Setting | Parameter | False alarms per 1000 windows |\n"
                  "|---|---|---|---|\n"
                  "| CUSUM | calibrated | h = 3.250 | 1.000 |\n"
                  "| EWMA | calibrated | L = 2.125 | 1.000 |\n"
                  "| CUSUM | uncalibrated | h = 5 | 0.853 |\n"
                  "| EWMA | uncalibrated | L = 3 | 0.887 |\n", encoding="utf-8")
    assert p4_app.committed_h_L(md) == (3.25, 2.125)


def test_committed_h_L_matches_committed_results_text():
    h, L = p4_app.committed_h_L()
    text = p4_app.DRIFT_RESULTS.read_text(encoding="utf-8")
    assert f"| CUSUM | calibrated | h = {h:.3f} |" in text
    assert f"| EWMA | calibrated | L = {L:.3f} |" in text


def test_committed_h_L_missing_rows_raises(tmp_path):
    md = tmp_path / "drift_results.md"
    md.write_text("no table here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        p4_app.committed_h_L(md)


# --- stream builder ----------------------------------------------------------
def synthetic_benign(n=16_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Header_Length": rng.uniform(0, 60, n).round(1),
        "Max": rng.uniform(40, 1500, n).round(1),
        "IAT": rng.uniform(0, 0.1, n),
        "Protocol Type": rng.choice([6.0, 17.0], n),
        data_prep.TARGET: np.zeros(n, dtype=int),
    })


FEATURES = ["Header_Length", "Max", "IAT"]


@pytest.mark.parametrize("seed", [0, 7])
def test_builder_at_factor_2_equals_experiment_builder(seed):
    df = synthetic_benign()
    got, got_f = p4_app.build_scenario_c_frame(df, FEATURES, seed, max_factor=2.0)
    want, want_f = de.build_scenario_c(df, FEATURES, seed)
    pd.testing.assert_frame_equal(got, want)
    np.testing.assert_array_equal(got_f, want_f)


def test_builder_factor_ramp_and_shape():
    df = synthetic_benign()
    frame, f = p4_app.build_scenario_c_frame(df, FEATURES, 0, max_factor=1.5)
    assert len(frame) == (de.N_BEFORE + de.N_AFTER) * de.WINDOW
    assert np.all(f[: de.N_BEFORE * de.WINDOW] == 1.0)
    assert f[-1] == pytest.approx(1.5) and f.max() == pytest.approx(1.5)
    assert frame["Protocol Type"].isin([6.0, 17.0]).all()  # unscaled features untouched


def test_builder_factor_1_leaves_rows_unchanged():
    df = synthetic_benign()
    frame, f = p4_app.build_scenario_c_frame(df, FEATURES, 3, max_factor=1.0)
    assert np.all(f == 1.0)
    idx = np.random.default_rng(3).choice(len(df), len(frame), replace=False)
    pd.testing.assert_frame_equal(frame, data_prep.quantise(df.iloc[idx].reset_index(drop=True)))


def test_builder_rejects_factor_below_one():
    with pytest.raises(ValueError):
        p4_app.build_scenario_c_frame(synthetic_benign(), FEATURES, 0, max_factor=0.9)


class FakeModel:
    """P(Mirai) rises with Header_Length, so scaling it produces drift."""

    def predict_proba(self, X):
        p = np.clip(X["Header_Length"].to_numpy() / 200.0, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def test_run_drift_stream_scores_both_detectors():
    df = synthetic_benign()
    ref_stream = de.window_logits(FakeModel().predict_proba(df.iloc[:10_000])[:, 1])
    ref = de.fit_reference(ref_stream)
    out = p4_app.run_drift_stream(FakeModel(), df, FEATURES, ref, h=4.875, L=2.96, seed=0, max_factor=2.0)
    assert len(out["stream"]) == de.N_BEFORE + de.N_AFTER
    for kind in ("cusum", "ewma"):
        r = out[kind]
        assert r["false_alarms"] == sum(a < de.N_BEFORE for a in r["alarms"])
        assert r["alarms_from_200"] == len(r["alarms"]) - r["false_alarms"]
        assert r["delay"] is not None and r["delay"] >= 0  # x2 Header_Length is a large shift
    assert out["cusum"]["param"] == 4.875 and out["ewma"]["param"] == 2.96


# --- other helpers -----------------------------------------------------------
def test_readme_check_rows_marks_pass_and_fail():
    expected = {"a": "0.5", "b": [1, 2]}
    rows = p4_app.readme_check_rows({"a": "0.5", "b": [1, 3]}, expected)
    assert [r[3] for r in rows] == ["PASS", "FAIL"]


def test_readme_check_rows_covers_every_documented_figure():
    measured = dict(leakage_audit.README_EXPECTED)
    rows = p4_app.readme_check_rows(measured)
    assert len(rows) == len(leakage_audit.README_EXPECTED) and all(r[3] == "PASS" for r in rows)


def test_summary_section_reads_pp2_plans():
    text = p4_app.summary_section("6. Planned for PP2")
    assert text.startswith("- ") and "## " not in text


def test_first_row_where():
    p = np.array([0.5, 0.995, 0.999])
    assert p4_app.first_row_where(p, lambda x: x > 0.99) == 1
    assert p4_app.first_row_where(p, lambda x: x > 2) == 0
