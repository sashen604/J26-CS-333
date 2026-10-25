"""Turn model output into the named scalar streams the drift detectors consume.

CLAUDE.md lists two monitored streams:
  (1) fused windowed anomaly score             -> implemented here
  (2) CSI-branch vs flow-branch disagreement   -> STUB, needs P1 CSI data and P2's AI model

The model is treated as a black box: anything with
predict_proba(X_flow, X_csi) -> (n, 2) array works, including the proxy.
Rows must be in time order for the stream to mean anything as a time series.
"""
from dataclasses import dataclass

import numpy as np

FUSED_ANOMALY_SCORE = "fused_windowed_anomaly_score"
BRANCH_DISAGREEMENT = "csi_flow_branch_disagreement"


@dataclass(frozen=True)
class ScalarStream:
    """A named 1-D series of scalars, one value per window."""
    name: str
    values: np.ndarray
    window: int
    step: int


def _window_means(scores, window, step):
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"Per-row scores must be 1-D, got shape {scores.shape}")
    if len(scores) < window:
        raise ValueError(f"Need at least {window} rows for one window, got {len(scores)}")
    starts = range(0, len(scores) - window + 1, step)
    return np.array([scores[s:s + window].mean() for s in starts])


def fused_windowed_anomaly_score(model, X_flow, X_csi=None, *, window, step=None):
    """Stream (1): mean P(attack) from model.predict_proba over sliding windows of rows.

    window: rows per window (required; no default, choose it per experiment).
    step:   rows between window starts; defaults to `window` (non-overlapping windows).
    An incomplete final window is dropped.
    """
    step = window if step is None else step
    proba = np.asarray(model.predict_proba(X_flow, X_csi))
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"predict_proba must return shape (n, 2), got {proba.shape}")
    values = _window_means(proba[:, 1], window, step)
    return ScalarStream(FUSED_ANOMALY_SCORE, values, window, step)


def branch_disagreement_score(model, X_flow, X_csi, *, window, step=None):
    """Stream (2): CSI-branch vs flow-branch disagreement. NOT IMPLEMENTED.

    STUB. It needs real CSI data (P1) and P2's AI model with a CSI branch.
    The proxy model has no CSI branch (branch_scores()['csi'] is None), so any
    value computed now would be fabricated. Implement once both exist, and agree
    the disagreement measure first (e.g. |p_csi - p_flow| per row, windowed).
    """
    raise NotImplementedError(
        "Stream (2) csi_flow_branch_disagreement needs P1 CSI data and P2's AI model; not available yet"
    )
