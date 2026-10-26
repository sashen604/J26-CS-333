"""Throwaway smoke test: check XGBoost, SHAP and LIME work together on this Python.

Run with:  .venv/Scripts/python p4_robustness/tests/smoke_deps.py
Not a pytest test (no test_ prefix), so pytest won't collect it.
"""
import sys

import lime
import lime.lime_tabular
import numpy as np
import shap
import xgboost
from sklearn.datasets import make_classification

print(f"Python {sys.version.split()[0]}, xgboost {xgboost.__version__}, shap {shap.__version__}")

# (a) Tiny XGBoost classifier on synthetic data
X, y = make_classification(n_samples=500, n_features=10, random_state=0)
feature_names = [f"f{i}" for i in range(X.shape[1])]
model = xgboost.XGBClassifier(n_estimators=50, max_depth=3, random_state=0)
model.fit(X, y)
print(f"(a) XGBoost trained, train accuracy = {model.score(X, y):.3f}")

# (b) SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print(f"(b) SHAP values shape = {np.shape(shap_values)}")

# (c) LIME on one instance
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X, feature_names=feature_names, class_names=["0", "1"], mode="classification", random_state=0
)
exp = lime_explainer.explain_instance(X[0], model.predict_proba, num_features=3)
print("(c) LIME top 3 features for row 0:")
for feature, weight in exp.as_list():
    print(f"    {feature:25} {weight:+.4f}")

print("SMOKE TEST PASSED")
