"""Tests for xai/explainers.py and xai/metrics.py on small synthetic data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from proxy_model import ProxyFusionModel
from xai.explainers import lime_explain, make_lime_explainer, shap_explain
from xai.metrics import (UPPER_BOUNDS, exai_repeatability, explanation_stability, jaccard,
                         perturb_row, spearman, top_k)

FEATURES = ["Header_Length", "IAT", "Rate", "Max", "Min"]


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=400, n_features=5, n_informative=3, random_state=0)
    X = pd.DataFrame(np.abs(X) * 10, columns=FEATURES)
    return X, pd.Series(y)


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return ProxyFusionModel.train(X, y, n_estimators=30, max_depth=3)


class SpyModel:
    """Wraps a model and counts predict_proba calls, to prove LIME uses the interface."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0
        self.feature_names = inner.feature_names
        self._check_flow = inner._check_flow

    def predict_proba(self, X_flow, X_csi=None):
        self.calls += 1
        return self.inner.predict_proba(X_flow, X_csi)


# ---- SHAP ----

def test_shap_shape_and_names(model, data):
    X, _ = data
    r = shap_explain(model, X.iloc[:20])
    assert r.values.shape == (20, 5)
    assert r.feature_names == FEATURES


def test_shap_is_additive_in_log_odds_toward_mirai(model, data):
    X, _ = data
    r = shap_explain(model, X.iloc[:50])
    log_odds = r.base_value + r.values.sum(axis=1)
    p = model.predict_proba(X.iloc[:50])[:, 1]
    assert np.allclose(1 / (1 + np.exp(-log_odds)), p, atol=1e-5)


def test_shap_reorders_columns_to_the_schema(model, data):
    X, _ = data
    a = shap_explain(model, X.iloc[:5]).values
    b = shap_explain(model, X.iloc[:5][FEATURES[::-1]].assign(extra=1.0)).values
    assert np.allclose(a, b)


# ---- LIME ----

def test_lime_goes_through_predict_proba(model, data):
    X, _ = data
    spy = SpyModel(model)
    lime_explain(spy, X.iloc[[0]], X, num_samples=200)
    assert spy.calls >= 1


def test_lime_returns_full_weights_and_sorted_top_k(model, data):
    X, _ = data
    r = lime_explain(model, X.iloc[[0]], X, k=3, num_samples=500)
    assert r.weights.shape == (5,)
    assert len(r.top_k) == 3
    mags = [abs(w) for _, w in r.top_k]
    assert mags == sorted(mags, reverse=True)
    assert all(f in FEATURES for f, _ in r.top_k)


def test_lime_is_deterministic_with_a_seed(model, data):
    X, _ = data
    expl = make_lime_explainer(model, X)
    a = lime_explain(model, X.iloc[[3]], X, explainer=expl, num_samples=300, seed=7).weights
    b = lime_explain(model, X.iloc[[3]], X, explainer=expl, num_samples=300, seed=7).weights
    assert np.array_equal(a, b)


def test_lime_rejects_more_than_one_row(model, data):
    X, _ = data
    with pytest.raises(ValueError):
        lime_explain(model, X.iloc[:2], X)


# ---- perturbations ----

def row_frame():
    return pd.DataFrame({"Header_Length": [59.0], "Time_To_Live": [250.0], "IAT": [0.01],
                         "Rate": [0.0], "Protocol Type": [6.0], "TCP": [1.0], "is_mirai": [0],
                         "Label": pd.array(["BenignTraffic"], dtype="string")})


def test_perturb_row_is_valid():
    rng = np.random.default_rng(0)
    base = row_frame()
    for _ in range(50):
        p = perturb_row(base, rng, rel_eps=0.2)
        assert p["Protocol Type"].iloc[0] == 6.0          # protocol code untouched
        assert p["TCP"].iloc[0] == 1.0                    # 0-1 share untouched
        assert p["is_mirai"].iloc[0] == 0                 # label untouched
        assert p["Label"].iloc[0] == "BenignTraffic"      # text columns skipped, not an error
        assert p["Rate"].iloc[0] == 0.0                   # multiplicative: zero stays zero
        assert (p.select_dtypes("number") >= 0).all().all()
        for col, bound in UPPER_BOUNDS.items():
            assert p[col].iloc[0] <= bound
        hl = p["Header_Length"].iloc[0]
        assert hl == round(hl, 1)                         # quantised feature on the 0.1 grid


def test_perturb_row_changes_continuous_features_within_eps():
    rng = np.random.default_rng(1)
    base = row_frame()
    p = perturb_row(base, rng, rel_eps=0.05)
    ratio = p["IAT"].iloc[0] / base["IAT"].iloc[0]
    assert 0.95 <= ratio <= 1.05 and ratio != 1.0


def test_perturb_row_is_reproducible_with_a_seed():
    a = perturb_row(row_frame(), np.random.default_rng(3))
    b = perturb_row(row_frame(), np.random.default_rng(3))
    assert a.equals(b)


# ---- stability metric ----

def test_top_k_breaks_ties_by_index():
    assert top_k([0.0, -3.0, 3.0, 1.0, 0.0], 3) == {1, 2, 3}
    assert top_k([0.0, 0.0, 0.0], 2) == {0, 1}


def test_jaccard_and_spearman():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert jaccard(set(), set()) == 1.0
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_constant_explainer_is_perfectly_stable():
    s = explanation_stability(lambda r: np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5]), row_frame(), k=3)
    assert s == {"jaccard": 1.0, "spearman": 1.0, "intersection": 1.0}


def test_unstable_explainer_scores_below_one():
    rng = np.random.default_rng(0)
    s = explanation_stability(lambda r: rng.normal(size=20), row_frame(), n_perturbations=10, k=5)
    assert s["jaccard"] < 1.0 and s["intersection"] < 1.0
    assert abs(s["spearman"]) < 0.5


def test_intersection_counts_features_kept_in_every_run():
    outputs = iter([np.array([5, 4, 3, 0, 0.0]),     # unperturbed: top-3 {0, 1, 2}
                    np.array([5, 4, 0, 3, 0.0]),     # {0, 1, 3}
                    np.array([5, 0, 4, 3, 0.0])])    # {0, 2, 3}
    s = explanation_stability(lambda r: next(outputs), row_frame(), n_perturbations=2, k=3)
    assert s["intersection"] == pytest.approx(1 / 3)  # only feature 0 survives every run
    assert s["jaccard"] == pytest.approx(0.5)         # each run shares 2 of 4 with the original


def test_stability_with_real_explainers_runs(model, data):
    X, _ = data
    row = X.iloc[[0]]
    s1 = explanation_stability(lambda r: shap_explain(model, r).values[0], row, n_perturbations=3, k=3)
    expl = make_lime_explainer(model, X)
    s2 = explanation_stability(
        lambda r: lime_explain(model, r, X, explainer=expl, num_samples=300).weights,
        row, n_perturbations=3, k=3)
    for s in (s1, s2):
        assert 0.0 <= s["jaccard"] <= 1.0 and 0.0 <= s["intersection"] <= 1.0


def test_keep_excludes_rejected_perturbations_and_full_metrics_are_unchanged():
    a = np.array([5, 4, 3, 0, 0.0])
    b = np.array([0, 0, 3, 4, 5.0])  # completely different top-3
    calls = []

    def fn(r):
        calls.append(r)
        return a if len(calls) == 1 or len(calls) % 2 == 0 else b

    def plain_fn(r, it=iter([a] + [a, b] * 5)):
        return next(it)

    full = explanation_stability(plain_fn, row_frame(), n_perturbations=4, k=3)
    s = explanation_stability(fn, row_frame(), n_perturbations=4, k=3,
                              keep=lambda xp: len(calls) % 2 == 0)  # keeps the runs that returned `a`
    assert {m: s[m] for m in ("jaccard", "spearman", "intersection")} == full
    assert s["n_kept"] == 2
    assert s["kept_jaccard"] == 1.0 and s["kept_intersection"] == 1.0


def test_keep_with_nothing_kept_gives_nan():
    s = explanation_stability(lambda r: np.arange(5.0), row_frame(), n_perturbations=3, k=2,
                              keep=lambda xp: False)
    assert s["n_kept"] == 0
    assert np.isnan(s["kept_jaccard"]) and np.isnan(s["kept_intersection"])
    assert s["jaccard"] == 1.0


def test_exai_repeatability():
    assert exai_repeatability(lambda r: np.array([5, 4, 3, 2, 1.0]), row_frame(), n_runs=3, k=2) == 1.0
    outs = iter([np.array([5, 4, 0, 0.0]), np.array([5, 0, 4, 0.0]), np.array([5, 0, 0, 4.0])])
    assert exai_repeatability(lambda r: next(outs), row_frame(), n_runs=3, k=2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        exai_repeatability(lambda r: np.zeros(3), pd.concat([row_frame(), row_frame()]))


def test_lime_without_seed_draws_fresh_samples(model, data):
    X, _ = data
    expl = make_lime_explainer(model, X, seed=0)
    a = lime_explain(model, X.iloc[[3]], X, explainer=expl, num_samples=300, seed=None).weights
    b = lime_explain(model, X.iloc[[3]], X, explainer=expl, num_samples=300, seed=None).weights
    assert not np.array_equal(a, b)


def test_shap_is_perfectly_repeatable(model, data):
    X, _ = data
    assert exai_repeatability(lambda r: shap_explain(model, r).values[0], X.iloc[[0]], n_runs=3, k=3) == 1.0


def test_stability_rejects_more_than_one_row(data):
    X, _ = data
    with pytest.raises(ValueError):
        explanation_stability(lambda r: np.zeros(5), X.iloc[:2])
