"""Tests for preflight.py. They use temporary files and a tiny synthetic model, so no download is needed."""
import json

import numpy as np
import pandas as pd
import pytest

import data_prep
import preflight
from proxy_model import ProxyFusionModel


def test_check_venv_passes_only_for_the_repo_venv(tmp_path):
    assert preflight.check_venv(prefix=preflight.VENV_DIR).ok
    bad = preflight.check_venv(prefix=tmp_path)
    assert not bad.ok and preflight.VENV_PYTHON in bad.fix


def test_read_requirements_parses_pins_and_skips_comments(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==2.5.4\n# a comment\n\nscikit-learn==1.9.1  # inline\n")
    assert preflight.read_requirements(req) == [("numpy", "2.5.4"), ("scikit-learn", "1.9.1")]


def test_import_names_cover_renamed_packages():
    assert preflight.import_name("scikit-learn") == "sklearn"
    assert preflight.import_name("python-dateutil") == "dateutil"
    assert preflight.import_name("lazy-loader") == "lazy_loader"
    assert preflight.import_name("numpy") == "numpy"


def test_check_packages_passes_on_installed_pin(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(f"numpy=={np.__version__}\n")
    assert preflight.check_packages(req).ok


def test_check_packages_fails_on_missing_package_and_wrong_pin(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==0.0.1\nno-such-package-p4==1.0\n")
    c = preflight.check_packages(req)
    assert not c.ok
    assert "numpy" in c.detail and "no-such-package-p4 does not import" in c.detail
    assert "pip install -r p4_robustness/requirements.txt" in c.fix


def test_check_packages_on_real_requirements_file():
    assert preflight.check_packages().ok


def test_check_data_files(tmp_path):
    (tmp_path / "train.parquet").write_bytes(b"x" * 10)
    (tmp_path / "test.parquet").write_bytes(b"x" * 7)
    checks = preflight.check_data_files(tmp_path, {"train": 10, "test": 5, "validation": 3})
    assert [c.ok for c in checks] == [True, False, False]
    assert "expected 5" in checks[1].detail
    assert checks[2].fix == preflight.run_cmd("data_prep.py")
    assert "data_prep.py" in checks[1].fix


def write_split_and_model(tmp_path, features):
    """A tiny test split and a model trained on `features`, saved under tmp_path."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.random(40) for f in ["Rate", "IAT", "Number"]})
    df["attack_class"] = ["Benign", "Mirai"] * 20
    df.to_parquet(data_prep.split_path("test", tmp_path))
    y = (df["attack_class"] == "Mirai").astype(int)
    model = ProxyFusionModel.train(df[features], y, n_estimators=3)
    model.save(tmp_path / "m.json", tmp_path / "s.json")
    return tmp_path / "m.json", tmp_path / "s.json"


def test_check_model_passes_on_matching_features(tmp_path):
    m, s = write_split_and_model(tmp_path, ["Rate", "IAT"])
    c = preflight.check_model(m, s, tmp_path)
    assert c.ok, c.detail


def test_check_model_fails_on_feature_mismatch(tmp_path):
    m, s = write_split_and_model(tmp_path, ["Rate"])
    c = preflight.check_model(m, s, tmp_path)
    assert not c.ok and "IAT" in c.detail
    assert c.fix == preflight.run_cmd("proxy_model.py")


def test_check_model_fails_on_excluded_feature_in_schema(tmp_path):
    m, s = write_split_and_model(tmp_path, ["Rate", "IAT"])
    s.write_text(json.dumps({"feature_names": ["Rate", "IAT", "Number"]}))
    c = preflight.check_model(m, s, tmp_path)
    assert not c.ok and "Number" in c.detail


def test_check_model_fails_when_files_missing(tmp_path):
    c = preflight.check_model(tmp_path / "m.json", tmp_path / "s.json", tmp_path)
    assert not c.ok and "missing" in c.detail
    assert c.fix == preflight.run_cmd("proxy_model.py")


def test_check_model_fails_on_corrupt_model(tmp_path):
    m, s = write_split_and_model(tmp_path, ["Rate", "IAT"])
    m.write_text("not a model")
    c = preflight.check_model(m, s, tmp_path)
    assert not c.ok and "does not load" in c.detail


def test_check_results_on_committed_files():
    assert all(c.ok for c in preflight.check_results())
    assert preflight.check_plot().ok


def test_check_results_flags_missing_and_unparseable(tmp_path):
    (tmp_path / "drift_results.md").write_text("# nothing the demo can read\n## Scenario C\n")
    checks = {c.name: c for c in preflight.check_results(tmp_path)}
    assert not checks["results: drift_results.md"].ok
    assert not checks["results: xai_results.md"].ok
    assert "git checkout --" in checks["results: xai_results.md"].fix
    assert not preflight.check_plot(tmp_path).ok


def test_main_prints_ready_only_when_all_pass(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "run_all", lambda: [preflight.Check("a", True)])
    assert preflight.main() == 0
    assert "READY FOR DEMO" in capsys.readouterr().out
    monkeypatch.setattr(preflight, "run_all",
                        lambda: [preflight.Check("a", True), preflight.Check("b", False, "x", "do y")])
    assert preflight.main() == 1
    out = capsys.readouterr().out
    assert "READY FOR DEMO" not in out and "fix: do y" in out


@pytest.mark.parametrize("script", ["data_prep.py", "proxy_model.py"])
def test_run_cmd_uses_venv_python(script):
    assert preflight.run_cmd(script) == f"{preflight.VENV_PYTHON} p4_robustness/{script}"
