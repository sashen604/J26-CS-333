"""Tests for drift/streams.py, using a stub model with known outputs."""
import numpy as np
import pandas as pd
import pytest

from drift.cusum import CUSUM
from drift.streams import (BRANCH_DISAGREEMENT, FUSED_ANOMALY_SCORE, ScalarStream,
                           branch_disagreement_score, fused_windowed_anomaly_score)


class StubModel:
    """Black-box stand-in: P(attack) is read from column 'p'."""

    def __init__(self):
        self.seen_csi = "unset"

    def predict_proba(self, X_flow, X_csi=None):
        self.seen_csi = X_csi
        p = X_flow["p"].to_numpy()
        return np.column_stack([1 - p, p])


@pytest.fixture
def X():
    return pd.DataFrame({"p": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5]})


def test_non_overlapping_window_means(X):
    s = fused_windowed_anomaly_score(StubModel(), X, window=3)
    assert isinstance(s, ScalarStream)
    assert s.name == FUSED_ANOMALY_SCORE
    assert (s.window, s.step) == (3, 3)
    # windows [0,.2,.4] and [.6,.8,1.0]; the incomplete last window [.5] is dropped
    assert np.allclose(s.values, [0.2, 0.8])


def test_sliding_windows_with_step(X):
    s = fused_windowed_anomaly_score(StubModel(), X, window=2, step=1)
    assert np.allclose(s.values, [0.1, 0.3, 0.5, 0.7, 0.9, 0.75])


def test_output_is_one_dimensional_and_feeds_cusum(X):
    s = fused_windowed_anomaly_score(StubModel(), X, window=2, step=1)
    assert s.values.ndim == 1
    d = CUSUM(s.name).fit_reference(s.values)
    assert len(d.run(s.values)) == len(s.values)


def test_x_csi_is_passed_through_to_model(X):
    m = StubModel()
    fused_windowed_anomaly_score(m, X, window=2)
    assert m.seen_csi is None


def test_window_validation(X):
    with pytest.raises(ValueError):
        fused_windowed_anomaly_score(StubModel(), X, window=0)
    with pytest.raises(ValueError):
        fused_windowed_anomaly_score(StubModel(), X, window=2, step=0)
    with pytest.raises(ValueError):
        fused_windowed_anomaly_score(StubModel(), X, window=100)


def test_bad_predict_proba_shape_is_rejected(X):
    class OneColumn:
        def predict_proba(self, X_flow, X_csi=None):
            return np.zeros((len(X_flow), 1))
    with pytest.raises(ValueError):
        fused_windowed_anomaly_score(OneColumn(), X, window=2)


def test_works_with_the_real_proxy_model_interface(X):
    from proxy_model import ProxyFusionModel
    rng = np.random.default_rng(0)
    Xf = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = (Xf["a"] > 0).astype(int)
    model = ProxyFusionModel.train(Xf, y, n_estimators=5, max_depth=2)
    s = fused_windowed_anomaly_score(model, Xf, window=10)
    assert s.values.shape == (6,)
    assert ((s.values >= 0) & (s.values <= 1)).all()


def test_stream_two_is_a_stub_and_refuses(X):
    assert BRANCH_DISAGREEMENT == "csi_flow_branch_disagreement"
    with pytest.raises(NotImplementedError):
        branch_disagreement_score(StubModel(), X, X_csi=None, window=2)
