"""Explanation stability: perturbation stability (main metric) and E-XAI repeatability.

CITATIONS
---------
- Perturbation stability: per Gan, Mao, Zhang, Ji, Pu, Han, Yin and Wang, '"Is your
  explanation stable?": A Robustness Evaluation Framework for Feature Attribution',
  ACM CCS 2022: the stability of an explanation on noisy inputs sampled from the
  neighbourhood of the original input (10 noisy inputs per input by default, Sec. 5.1.3).
  That paper's own stability metric is mstd (the mean standard deviation of attribution
  scores across noisy inputs, Sec. 5.1.3), which is NOT computed here.
- The `intersection` score uses E-XAI's formula (Arreche et al., IEEE Access 2024,
  Sec. C.4, PDF p. 8), adapted to perturbed inputs.
- `jaccard` and `spearman` are our additions.
- exai_repeatability() is E-XAI's original stability: the same formula over N repeated
  runs on the UNCHANGED row.

DEFINITION (perturbation stability)
-----------------------------------
For one row x with attribution vector a_0 (one value per feature; SHAP or LIME):
1. Make N perturbed copies x_1..x_N with perturb_row() (small, valid; see below) and
   explain each, giving a_1..a_N. The explainer's own randomness is held fixed (LIME
   uses the same seed every time), so only the input changes.
2. T_j = top-k features of a_j by |attribution| (ties broken by feature order).
3. Report, for that row:
   - jaccard     = mean over j of |T_0 ∩ T_j| / |T_0 ∪ T_j|
   - spearman    = mean over j of Spearman rank correlation of a_0 and a_j over ALL
                   features (signed values; NaN if either vector is constant)
   - intersection = |T_0 ∩ T_1 ∩ ... ∩ T_N| / k
All three are 1.0 when the explanation does not change.
Optionally, perturbations failing a `keep` test (e.g. those that flip the model's predicted
class) are excluded and the three numbers are recomputed over the kept ones ("kept_*").

E-XAI REPEATABILITY (Arreche et al., Sec. C.4, PDF p. 8)
--------------------------------------------------------
Explain the UNCHANGED row N times (N = 3 in E-XAI) with the explainer's own randomness
left on, take the top-k set X_i of each run, and report |X_1 ∩ ... ∩ X_N| / k. This is E-XAI's
published stability. It measures run-to-run noise in the explainer, not input sensitivity.

VALID PERTURBATIONS
-------------------
Each perturbable feature is multiplied by an independent factor drawn uniformly from
[1 - rel_eps, 1 + rel_eps] (default 5%, our choice). Multiplicative, so zero stays zero and
non-negative stays non-negative. Not perturbed: protocol codes and 0-1 share/flag features,
whose values have fixed meanings. Known upper bounds are enforced by clipping, and
QUANTISED_FEATURES are re-rounded to the 0.1 grid (CLAUDE.md locked decision).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data_prep

PROTOCOL_CODES = {"Protocol Type"}
SHARE_FEATURES = {
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
}
NOT_PERTURBED = PROTOCOL_CODES | SHARE_FEATURES
UPPER_BOUNDS = {
    "Time_To_Live": 255.0,   # IP TTL limit
    "Header_Length": 60.0,   # test-split max; possibly the TCP header byte limit (see drift_results.md)
}
DEFAULT_REL_EPS = 0.05


def perturb_row(row, rng, rel_eps=DEFAULT_REL_EPS):
    """Return a perturbed copy of a 1-row DataFrame (see VALID PERTURBATIONS above)."""
    out = row.copy()
    for col in out.columns:
        if col in NOT_PERTURBED or col == data_prep.TARGET or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = out[col] * rng.uniform(1 - rel_eps, 1 + rel_eps)
        if col in UPPER_BOUNDS:
            out[col] = out[col].clip(upper=UPPER_BOUNDS[col])
    return data_prep.quantise(out)


def top_k(attributions, k):
    """Indices of the k largest |attributions|, ties broken by lower index."""
    a = np.asarray(attributions, dtype=float)
    return set(sorted(range(len(a)), key=lambda i: (-abs(a[i]), i))[:k])


def jaccard(s, t):
    return len(s & t) / len(s | t) if s | t else 1.0


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _summarise_runs(t0, runs, k):
    """jaccard / spearman / intersection over a list of (attributions, top-k set) runs."""
    if not runs:
        return {"jaccard": float("nan"), "spearman": float("nan"), "intersection": float("nan")}
    a0, t0set = t0
    jac = [jaccard(t0set, tj) for _, tj in runs]
    rho = [spearman(a0, aj) for aj, _ in runs]
    common = set(t0set)
    for _, tj in runs:
        common &= tj
    return {
        "jaccard": float(np.mean(jac)),
        "spearman": float(np.nanmean(rho)) if not np.all(np.isnan(rho)) else float("nan"),
        "intersection": len(common) / k,
    }


def explanation_stability(explain_fn, row, n_perturbations=10, k=5, rel_eps=DEFAULT_REL_EPS, seed=0,
                          keep=None):
    """Perturbation stability of explain_fn for one row. explain_fn(1-row DataFrame) -> 1-D attributions.

    Returns {'jaccard', 'spearman', 'intersection'} as defined in the module docstring. If
    `keep(perturbed_row) -> bool` is given, also returns 'kept_jaccard', 'kept_spearman',
    'kept_intersection' over only the kept perturbations (NaN if none), and 'n_kept'.
    The perturbations themselves are the same either way.
    """
    if len(row) != 1:
        raise ValueError(f"explanation_stability takes one row, got {len(row)}")
    rng = np.random.default_rng(seed)
    a0 = np.asarray(explain_fn(row), dtype=float)
    base = (a0, top_k(a0, k))
    runs, kept = [], []
    for _ in range(n_perturbations):
        xp = perturb_row(row, rng, rel_eps)
        aj = np.asarray(explain_fn(xp), dtype=float)
        runs.append((aj, top_k(aj, k)))
        if keep is not None and keep(xp):
            kept.append(runs[-1])
    out = _summarise_runs(base, runs, k)
    if keep is not None:
        out.update({f"kept_{m}": v for m, v in _summarise_runs(base, kept, k).items()})
        out["n_kept"] = len(kept)
    return out


def exai_repeatability(explain_fn, row, n_runs=3, k=5):
    """E-XAI stability: |X_1 ∩ ... ∩ X_N| / k over n_runs explanations of the UNCHANGED row.

    explain_fn must keep its own randomness on (e.g. LIME not reseeded per call).
    """
    if len(row) != 1:
        raise ValueError(f"exai_repeatability takes one row, got {len(row)}")
    common = None
    for _ in range(n_runs):
        t = top_k(np.asarray(explain_fn(row), dtype=float), k)
        common = t if common is None else common & t
    return len(common) / k
