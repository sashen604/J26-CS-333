"""Tests for experiments/make_figures.py: parsers read the committed files, and figures render."""
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


def test_leakage_rule_raises_when_figure_missing(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("nothing here", encoding="utf-8")
    with pytest.raises(ValueError):
        mf.leakage_rule(p)


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
