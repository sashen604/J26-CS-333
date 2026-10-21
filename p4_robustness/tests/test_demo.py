"""Tests for demo.py: committed-results parsers and part-failure handling. No data or model needed."""
import demo


def test_committed_scenario_c_reads_both_detectors():
    rows = demo.committed_scenario_c()
    assert set(rows) == {"CUSUM", "EWMA"}
    param, delay, fa, missed = rows["CUSUM"]
    assert param.startswith("h = ") and "±" in delay and missed.endswith("/20")


def test_committed_calibration_reads_calibrated_rows_only():
    rows = demo.committed_calibration()
    assert rows["CUSUM"][0].startswith("h = ") and rows["EWMA"][0].startswith("L = ")


def test_committed_shap_top_has_five_ranked_features():
    top = demo.committed_shap_top()
    assert len(top) == demo.TOP_N
    assert [v for _, v, _ in top] == sorted((v for _, v, _ in top), reverse=True)


def test_committed_repeatability_has_both_methods_and_classes():
    rows = demo.committed_repeatability()
    assert set(rows) == {(m, c) for m in ("SHAP", "LIME") for c in ("benign", "Mirai")}


def test_quick_settings_are_smaller_than_full():
    for field in ("calib_streams", "n_seeds", "n_summary", "n_repeat_rows"):
        assert getattr(demo.QUICK, field) < getattr(demo.FULL, field)


def test_failed_part_does_not_stop_the_demo(monkeypatch, capsys):
    ran = []

    def broken(settings, quick):
        raise RuntimeError("boom")

    def ok(settings, quick):
        ran.append(quick)

    monkeypatch.setattr(demo, "PARTS", (broken, ok))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert demo.main(["--quick"]) == 1
    out = capsys.readouterr().out
    assert ran == [True]
    assert "broken failed" in out and "RuntimeError: boom" in out


def test_wait_for_enter_survives_closed_stdin(monkeypatch):
    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    demo.wait_for_enter()
