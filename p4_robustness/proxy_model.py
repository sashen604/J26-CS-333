"""Proxy for P2's AI model, so Pillar 4 can work before the real model exists.

PURPOSE: this model exists to exercise the predict_proba interface and to serve
as the baseline for adversarial evaluation. Its accuracy MUST NOT be reported as
a detection result. Benign vs Mirai flood traffic is trivially separable on this
subset; see the Limitations section in data/README.md.

It is an XGBoost binary classifier (0 = benign, 1 = Mirai) trained on flow
features from the "CICIoT2023 subset (HF mirror)"; see data_prep.py for
provenance. Features that depend on the measurement window length are
excluded as window-size leakage (data_prep.EXCLUDED_FEATURES), and train()
refuses any of them. It is wrapped in the interface from
p4_robustness/CLAUDE.md:

    predict_proba(X_flow, X_csi=None) -> probabilities, shape (n, 2)
    branch_scores(X_flow, X_csi=None) -> per-branch Mirai scores

There is no CSI data yet, so this proxy has a flow branch only. X_csi is
accepted as an argument so callers can already use the final call shape, but
passing CSI data raises NotImplementedError rather than silently ignoring it.
When P1 delivers CSI, swap this class for P2's AI model behind the same interface.

Run `python p4_robustness/proxy_model.py` to train, evaluate on the held-out
splits and save the model to data/models/ (gitignored).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import data_prep

MODEL_DIR = data_prep.REPO_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "proxy_xgb.json"
SCHEMA_PATH = MODEL_DIR / "proxy_xgb_schema.json"

XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=0, n_jobs=-1)


class ProxyFusionModel:
    """Flow-only stand-in for P2's AI model, with a fixed feature schema."""

    def __init__(self, booster_model, feature_names):
        self.model = booster_model
        self.feature_names = list(feature_names)

    @classmethod
    def train(cls, X_flow, y, **params):
        if not isinstance(X_flow, pd.DataFrame):
            raise TypeError("X_flow must be a pandas DataFrame so the feature schema is fixed by column name")
        # Guard: refuse to train on any window-size-leakage feature (data_prep.EXCLUDED_FEATURES).
        leaked = [c for c in X_flow.columns if c in data_prep.EXCLUDED_FEATURES]
        if leaked:
            raise ValueError(f"Refusing to train on excluded (window-size leakage) features: {leaked}")
        model = xgboost.XGBClassifier(**{**XGB_PARAMS, **params})
        model.fit(X_flow, y)
        return cls(model, X_flow.columns)

    def _check_flow(self, X_flow):
        if not isinstance(X_flow, pd.DataFrame):
            raise TypeError("X_flow must be a pandas DataFrame")
        missing = [c for c in self.feature_names if c not in X_flow.columns]
        if missing:
            raise ValueError(f"X_flow is missing schema columns: {missing}")
        # Reorder to the training schema; extra columns (e.g. labels) are dropped.
        return X_flow[self.feature_names]

    @staticmethod
    def _check_csi(X_csi):
        if X_csi is not None:
            raise NotImplementedError("This proxy has no CSI branch yet; pass X_csi=None")

    def branch_scores(self, X_flow, X_csi=None):
        """Per-branch probability of Mirai. 'csi' is None until a CSI branch exists."""
        self._check_csi(X_csi)
        flow = self.model.predict_proba(self._check_flow(X_flow))[:, 1]
        return {"flow": flow, "csi": None}

    def predict_proba(self, X_flow, X_csi=None):
        """Columns: [P(benign), P(Mirai)]. With only a flow branch, fused = flow."""
        p_mirai = self.branch_scores(X_flow, X_csi)["flow"]
        return np.column_stack([1.0 - p_mirai, p_mirai])

    def predict(self, X_flow, X_csi=None, threshold=0.5):
        return (self.predict_proba(X_flow, X_csi)[:, 1] >= threshold).astype(int)

    def save(self, model_path=MODEL_PATH, schema_path=SCHEMA_PATH):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        Path(schema_path).write_text(json.dumps({"feature_names": self.feature_names}, indent=2))

    @classmethod
    def load(cls, model_path=MODEL_PATH, schema_path=SCHEMA_PATH):
        model = xgboost.XGBClassifier()
        model.load_model(model_path)
        feature_names = json.loads(Path(schema_path).read_text())["feature_names"]
        return cls(model, feature_names)


def evaluate(model, df):
    """Accuracy, precision, recall, F1 and confusion matrix on a filtered split (Mirai = positive)."""
    y_true = df[data_prep.TARGET].to_numpy()
    y_pred = model.predict(df)
    return {
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # rows = true [benign, mirai], columns = predicted [benign, mirai]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main():
    train = data_prep.load_split("train")
    # feature_columns() drops label columns and EXCLUDED_FEATURES (window-size leakage);
    # ProxyFusionModel.train() refuses to train if any of them gets back in.
    features = data_prep.feature_columns(train)
    model = ProxyFusionModel.train(train[features], train[data_prep.TARGET])
    model.save()
    print(f"Saved model to {MODEL_PATH} ({len(features)} features)")
    print("Dataset: CICIoT2023 subset (HF mirror), Mirai vs benign")
    for split in ("test", "validation"):
        m = evaluate(model, data_prep.load_split(split))
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        print(f"\n[{split}] n={m['n']:,}  accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}")
        print("                 pred benign  pred mirai")
        print(f"  true benign    {tn:11,}  {fp:10,}")
        print(f"  true mirai     {fn:11,}  {tp:10,}")


if __name__ == "__main__":
    main()
