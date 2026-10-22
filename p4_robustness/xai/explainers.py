"""SHAP and LIME wrappers around the Pillar 4 model (proxy now, P3's fusion model later).

Both explain P(Mirai), the second column of predict_proba(X_flow, X_csi).

- LIME goes entirely through the CLAUDE.md interface: it only calls
  model.predict_proba(X_flow) on a DataFrame with the model's fixed feature schema.
- SHAP uses shap.TreeExplainer, which REQUIRES the underlying tree model
  (model.model, the XGBClassifier); it cannot work through predict_proba. Input rows
  still pass through the model's schema check first. When P3's model replaces the
  proxy, TreeExplainer only applies if it is also a tree ensemble; otherwise a
  model-agnostic SHAP explainer (e.g. KernelExplainer on predict_proba) is needed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


@dataclass
class ShapResult:
    values: np.ndarray        # (n_rows, n_features)
    feature_names: list
    base_value: float


@dataclass
class LimeResult:
    weights: np.ndarray       # (n_features,) LIME weight per feature, 0 if LIME gave none
    feature_names: list
    top_k: list               # [(feature, weight)] sorted by |weight|, largest first


def make_shap_explainer(model):
    """TreeExplainer on the model's underlying XGBoost classifier (see module docstring)."""
    return shap.TreeExplainer(model.model)


def shap_explain(model, X, explainer=None):
    """SHAP TreeExplainer values for the rows of X.

    Values are in the model's raw output space (log-odds of Mirai). For this binary
    model, POSITIVE values push the prediction toward Mirai and NEGATIVE values toward
    benign. For each row, base_value + values.sum() equals the model's log-odds.
    Requires the underlying tree model, not predict_proba (see module docstring).
    """
    X = model._check_flow(X)
    explainer = explainer or make_shap_explainer(model)
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:  # some SHAP versions return one array per class; keep Mirai
        values = values[..., 1]
    base = np.atleast_1d(explainer.expected_value)[-1]
    return ShapResult(values, list(model.feature_names), float(base))


def make_lime_explainer(model, X_reference, seed=0):
    """LIME tabular explainer whose sampling statistics come from X_reference."""
    ref = model._check_flow(X_reference)
    return LimeTabularExplainer(
        ref.to_numpy(), feature_names=list(model.feature_names), class_names=["benign", "Mirai"],
        mode="classification", random_state=seed)


def _reseed_lime(explainer, seed):
    """LIME keeps separate random states in the explainer, its LimeBase and its discretizer
    (used to sample values inside bins). Reset all three so one seed fixes the explanation."""
    rs = np.random.RandomState(seed)
    explainer.random_state = rs
    explainer.base.random_state = rs
    if getattr(explainer, "discretizer", None) is not None:
        explainer.discretizer.random_state = rs


def lime_explain(model, X_row, X_reference, k=5, explainer=None, num_samples=5000, seed=0):
    """LIME explanation of P(Mirai) for one row. Returns all feature weights and the top-k.

    Uses only model.predict_proba (the CLAUDE.md interface). LIME is asked for weights on
    every feature, so `weights` is a full vector, and the top-k come from it by |weight|.
    `seed` fixes LIME's own sampling so repeated calls on the same row agree. With
    seed=None the explainer is NOT reseeded, so each call draws fresh LIME samples
    (LIME's normal random behaviour, needed for E-XAI repeatability).
    """
    features = list(model.feature_names)
    row = model._check_flow(X_row)
    if len(row) != 1:
        raise ValueError(f"lime_explain explains one row, got {len(row)}")
    if explainer is None:
        explainer = make_lime_explainer(model, X_reference, seed)
    if seed is not None:
        _reseed_lime(explainer, seed)

    def predict_fn(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=features))

    exp = explainer.explain_instance(row.to_numpy()[0], predict_fn, labels=(1,),
                                     num_features=len(features), num_samples=num_samples)
    weights = np.zeros(len(features))
    for idx, w in exp.as_map()[1]:
        weights[idx] = w
    order = sorted(range(len(features)), key=lambda i: (-abs(weights[i]), i))[:k]
    return LimeResult(weights, features, [(features[i], float(weights[i])) for i in order])
