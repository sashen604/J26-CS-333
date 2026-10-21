"""Tests for experiments/leakage_audit.py. Small synthetic frames, so no download is needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import data_prep  # noqa: E402
import leakage_audit as la  # noqa: E402


def make_raw(n_benign=8, n_mirai=6):
    """Raw-looking frame: benign windows of 10 packets (0.1 grid), Mirai of 100 (0.01 grid)."""
    rng = np.random.default_rng(0)
    number = np.r_[np.full(n_benign, 10), np.full(n_mirai, 100)]
    avg = np.r_[rng.integers(600, 1500, n_benign) / 10, rng.integers(6000, 15000, n_mirai) / 100]
    df = pd.DataFrame({
        "Number": number,
        "AVG": avg,
        "Tot size": avg,
        "Tot sum": number * avg,
        "attack_class": ["Benign"] * n_benign + ["Mirai"] * n_mirai,
    })
    for f in la.COUNT_FEATURES:
        df[f] = rng.integers(0, number + 1)
    for f in data_prep.QUANTISED_FEATURES:
        if f != "AVG":
            df[f] = np.r_[rng.integers(0, 10, n_benign) / 10, rng.integers(0, 100, n_mirai) / 100]
    # Mirai rows must be off the 0.1 grid somewhere, so set one feature explicitly.
    df.loc[df["attack_class"] == "Mirai", "TCP"] = 0.37
    return data_prep.filter_mirai_vs_benign(df)


def test_number_distribution_reports_window_sizes():
    df = make_raw()
    df.loc[0, "Number"] = 4
    out = la.number_distribution(df)
    assert out["benign_share_10"] == pytest.approx(7 / 8)
    assert out["mirai_share_100"] == 1.0
    assert (out["n_benign"], out["n_mirai"]) == (8, 6)


def test_totsum_ratio_is_one_when_tot_sum_is_number_times_avg():
    out = la.totsum_ratio_percentiles(make_raw())
    assert np.allclose(out["benign"], 1.0) and np.allclose(out["mirai"], 1.0)


def test_count_feature_bounds():
    df = make_raw()
    df.loc[0, "ack_count"] = 11  # one benign row above its Number of 10
    out = la.count_feature_bounds(df)
    assert out["ack_count"]["le_number_share"] == pytest.approx(13 / 14)
    assert out["ack_count"]["max_benign"] == 11
    assert out["rst_count"]["le_number_share"] == 1.0


def test_tot_size_equals_avg_share():
    df = make_raw()
    assert la.tot_size_equals_avg_share(df) == 1.0
    df.loc[0, "Tot size"] += 1
    assert la.tot_size_equals_avg_share(df) == pytest.approx(13 / 14)


def test_off_grid_rows_detects_hundredths_but_not_float_noise():
    df = pd.DataFrame({"a": [0.3, 0.1 + 0.2, 0.37, 12.0], "b": [1.0, 2.5, 0.0, 0.05]})
    assert la.off_grid_rows(df, ["a", "b"]).tolist() == [False, False, True, True]


def test_grid_rule_perfect_before_and_majority_after_quantising():
    df = make_raw()
    assert la.grid_rule_accuracy(df) == 1.0
    assert la.grid_rule_accuracy(data_prep.quantise(df)) == pytest.approx(la.majority_baseline(df))
    assert la.majority_baseline(df) == pytest.approx(8 / 14)


def test_number_stump_separates_10_from_100():
    df = make_raw()
    acc, threshold = la.number_stump_accuracy(df, df)
    assert acc == 1.0
    assert 10 < threshold < 100


def expected_like_readme():
    """A results dict that, formatted, equals README_EXPECTED."""
    counts = {f: {"le_number_share": 1.0,
                  "max_benign": la.README_EXPECTED["count_max_benign"][f],
                  "max_mirai": la.README_EXPECTED["count_max_mirai"][f]} for f in la.COUNT_FEATURES}
    return {
        "number": {"benign_share_10": 19994 / 19999, "mirai_share_100": 15075 / 15147},
        "stump": (0.99980, 10.5),
        "totsum_ratio": {"benign": [1.0, 1.0, 1.0], "mirai": [1.0, 1.0, 1.0]},
        "counts": counts,
        "tot_size_equals_avg": 1.0,
        "benign_rows_on_grid": (19995, 19999),
        "off_grid_benign_numbers": [4, 6, 8, 9],
        "grid_before": 0.81338,
        "grid_after": 0.56903,
        "majority": 0.56903,
    }


def test_compare_with_readme_passes_on_documented_values():
    assert la.compare_with_readme(expected_like_readme()) == []


def test_compare_with_readme_flags_the_old_train_split_figure():
    r = expected_like_readme()
    r["number"]["benign_share_10"] = 0.99950  # the train-split value the README used to show
    assert la.compare_with_readme(r) == [("number_benign_share_10", "99.95%", "99.97%")]


def test_run_audit_on_synthetic_splits(tmp_path):
    df = make_raw()
    df.drop(columns=[data_prep.TARGET]).to_parquet(data_prep.split_path("test", tmp_path))
    df.drop(columns=[data_prep.TARGET]).to_parquet(data_prep.split_path("train", tmp_path))
    r = la.run_audit(tmp_path)
    assert r["grid_before"] == 1.0
    assert r["benign_rows_on_grid"] == (8, 8)
    assert r["off_grid_benign_numbers"] == []
    la.print_report(r)  # must not raise


def test_load_raw_is_not_quantised(tmp_path):
    df = make_raw()
    df.drop(columns=[data_prep.TARGET]).to_parquet(data_prep.split_path("test", tmp_path))
    out = la.load_raw("test", tmp_path)
    assert (out["TCP"] == 0.37).sum() == 6


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        la.load_raw("test", tmp_path)
