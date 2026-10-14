"""Tests for proxy_model.py. They train on small synthetic data, so no download is needed."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import data_prep
from proxy_model import ProxyFusionModel, evaluate


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    return X, pd.Series(y)


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return ProxyFusionModel.train(X, y, n_estimators=20, max_depth=3)


def test_predict_proba_shape_and_rows_sum_to_one(model, data):
    X, _ = data
    p = model.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()


def test_x_csi_is_optional(model, data):
    X, _ = data
    assert np.array_equal(model.predict_proba(X), model.predict_proba(X, X_csi=None))


def test_passing_csi_raises_until_csi_branch_exists(model, data):
    X, _ = data
    with pytest.raises(NotImplementedError):
        model.predict_proba(X, X_csi=np.zeros((len(X), 3)))
    with pytest.raises(NotImplementedError):
        model.branch_scores(X, X_csi=np.zeros((len(X), 3)))


def test_branch_scores_flow_matches_fused_and_csi_is_none(model, data):
    X, _ = data
    scores = model.branch_scores(X)
    assert scores["csi"] is None
    assert np.allclose(scores["flow"], model.predict_proba(X)[:, 1])


def test_schema_missing_column_raises(model, data):
    X, _ = data
    with pytest.raises(ValueError, match="f2"):
        model.predict_proba(X.drop(columns=["f2"]))


def test_schema_rejects_non_dataframe(model, data):
    X, _ = data
    with pytest.raises(TypeError):
        model.predict_proba(X.to_numpy())


def test_column_order_and_extra_columns_do_not_matter(model, data):
    X, _ = data
    shuffled = X[X.columns[::-1]].assign(extra=1.0)
    assert np.allclose(model.predict_proba(shuffled), model.predict_proba(X))


def test_save_and_load_round_trip(model, data, tmp_path):
    X, _ = data
    model.save(tmp_path / "m.json", tmp_path / "schema.json")
    loaded = ProxyFusionModel.load(tmp_path / "m.json", tmp_path / "schema.json")
    assert loaded.feature_names == model.feature_names
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))


def test_evaluate_reports_all_metrics(model, data):
    X, y = data
    df = X.assign(**{data_prep.TARGET: y.to_numpy()})
    m = evaluate(model, df)
    assert set(m) == {"n", "accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert np.array(m["confusion_matrix"]).sum() == len(df)
    assert 0.0 <= m["f1"] <= 1.0
