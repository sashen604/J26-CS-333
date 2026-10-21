"""Tests for experiments/make_figures.py: parsers read the committed files, and figures render.

Tests that need the raw test split (figure 2) are skipped when it has not been downloaded.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import make_figures as mf  # noqa: E402


def test_leakage_rule_reads_data_readme():
    before, after = mf.leakage_rule()
    assert 0 < after < before < 1


def test_number_shares_reads_counts_that_agree_with_percentages():
    s = mf.number_shares()
    for pct, n, total in s.values():
        assert round(100 * n / total, 2) == pct


def test_calibration_has_four_rows_and_target():
    cal = mf.calibration()
    assert {k[1] for k in cal} == {"calibrated", "uncalibrated"}
    assert mf.target_far() > 0


def test_scenario_c_delays_and_seeds():
    d = mf.scenario_c_delays()
    assert set(d) == {"CUSUM", "EWMA"}
    assert all(mean > 0 and std >= 0 for _, mean, std, _ in d.values())
    assert mf.n_seeds() > 1


def test_shap_top10_is_sorted_and_row_count_parsed():
    rows = mf.shap_top10()
    vals = [v for _, v, _ in rows]
    assert vals == sorted(vals, reverse=True)
    assert mf.shap_rows() > 0


def test_repeatability_within_unit_interval():
    rep = mf.repeatability()
    assert all(0 <= m <= 1 for m, _ in rep.values())


def test_mean_std_rejects_other_text():
    assert mf.mean_std("16.65 ± 10.38") == (16.65, 10.38)
    with pytest.raises(ValueError):
        mf.mean_std("n/a")


def number_frame():
    import pandas as pd
    # benign: 4 x Number 10, 1 x Number 4; Mirai: 3 x Number 100, 1 x Number 10, 1 x Number 57
    return pd.DataFrame({"Number": [10, 10, 10, 10, 4, 100, 100, 100, 10, 57],
                         "is_mirai": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})


def test_number_groups_counts_10_100_and_other_per_class():
    g = mf.number_groups(number_frame())
    assert g["benign"] == {"10": 4, "100": 0, "Other": 1, "total": 5}
    assert g["Mirai"] == {"10": 1, "100": 3, "Other": 1, "total": 5}


def test_check_headline_passes_and_raises():
    g = mf.number_groups(number_frame())
    mf.check_headline(g, {"benign": (80.0, 4, 5), "Mirai": (60.0, 3, 5)})
    with pytest.raises(ValueError, match="benign Number = 10"):
        mf.check_headline(g, {"benign": (99.97, 19994, 19999), "Mirai": (60.0, 3, 5)})


@pytest.mark.skipif(not (mf.REPO_ROOT / "data" / "ciciot2023_hf" / "test.parquet").exists(),
                    reason="needs the raw test split (python p4_robustness/data_prep.py)")
def test_raw_number_groups_match_data_readme():
    g = mf.raw_number_groups()  # raises if the headline shares differ from data/README.md
    assert g["benign"]["total"] == mf.number_shares()["benign"][2]


def test_leakage_rule_raises_when_figure_missing(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("nothing here", encoding="utf-8")
    with pytest.raises(ValueError):
        mf.leakage_rule(p)


@pytest.mark.skipif(not (mf.REPO_ROOT / "data" / "ciciot2023_hf" / "test.parquet").exists(),
                    reason="figure 2 needs the raw test split (python p4_robustness/data_prep.py)")
def test_main_writes_six_16_9_white_pngs(tmp_path):
    from PIL import Image
    paths = mf.main(tmp_path)
    assert [p.name for p in paths] == ["leakage.png", "leakage_number.png", "drift_calibration.png",
                                       "drift_scenario_c_delay.png", "shap_top10.png", "repeatability.png"]
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        assert abs(w / h - 16 / 9) < 0.01
        assert img.getpixel((w - 2, 2)) == (255, 255, 255)
