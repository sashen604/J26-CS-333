# Pillar 4 progress summary (PP1)

All numbers below come from `drift_results.md`, `xai_results.md` (this folder) and
`data/README.md`. Results are on the **CICIoT2023 subset (HF mirror)**, not official CICIoT2023.

## 1. What Pillar 4 is

Pillar 4 evaluates the dual-layer IoT botnet detector for robustness, explainability and drift.
It covers drift detection (CUSUM, with EWMA as the required baseline), explanations (SHAP and
LIME), adversarial robustness and incident reporting. P2's AI model is treated as a black box.
Until it is available, an **XGBoost proxy model** stands in for it. The proxy's near-perfect
accuracy is not a detection result.

## 2. Data and its limits

- Source: an unofficial Hugging Face mirror (`lacg030175/CIC-IoT-2023`), already reduced by a third
  party (benign capped at 200,000 rows, 50,000 per attack type, 39 features).
- Its columns share names with official CICIoT2023 features but **not their values**. For example,
  `Number` reaches 100 here versus an official maximum of 15.
- Task: Mirai floods versus benign only. The classes are almost perfectly separable, so high
  accuracy here is not evidence of a strong detector.
- Replication on the official UNB data is **still pending**.

## 3. Methodology finding: window-size leakage

An audit of the features found that the capture window length differs by class: 10 packets for
benign, 100 for Mirai. `Number` is 10 for 99.97% of benign rows (19,994 of 19,999) and 100
for 99.52% of Mirai rows (15,075 of 15,147),
and a one-split rule on it alone reached 0.9998 test accuracy. Count features were bounded by
`Number`, and `Tot sum` equalled `Number × AVG` on every percentile checked.

- **Drop:** 7 window-dependent features (`Number`, `Tot sum`, `ack_count`, `rst_count`,
  `syn_count`, `fin_count`, `Tot size`).
- **Quantise:** benign means fall on a 0.1 grid and Mirai means on a 0.01 grid. Before rounding,
  99.98% of benign rows (19,995 of 19,999) were on the 0.1 grid. The 4 off-grid benign rows have
  `Number` = 4, 9, 8 and 6, so off-grid values follow window size, which supports the leakage
  explanation. The rule "off the 0.1
  grid → Mirai" scored 0.8134. After rounding 28 features to 0.1 for both classes, it scores 0.5690,
  which is the majority-class baseline.
- **Residual:** `Std` and `Variance` still depend on sample size. The full fix is to recompute the
  features from the PCAPs with a single window length.

## 4. Drift result: CUSUM vs EWMA

Both detectors monitor one scalar (the logit of the windowed P(attack)). Each was calibrated to
**1.000 false alarm per 1000 windows** on 150,000 benign windows (CUSUM h = 4.875, EWMA L = 2.960).
That makes their detection delays directly comparable.

- Scenarios A (abrupt) and B (gradual Mirai share): both detect almost immediately (B: 4.90 vs 4.75
  windows), mostly because the classes are so separable.
- **Scenario C (gradual covariate drift in benign traffic,** 5 features scaled ×1.0 → ×2.0):
  CUSUM **16.65 ± 10.38** windows, EWMA **17.00 ± 9.99** windows, 0/20 missed for both,
  0.30 ± 0.66 false alarms before the drift for both. **The two detectors are tied.**
- Caveat: scaling pushed 6.1% of `Header_Length` rows above its test maximum of 60, which may be
  physically invalid. All streams resample the same test rows, so the ± values likely understate
  the true uncertainty.

## 5. Explainability result: SHAP vs LIME

- **Global:** `Header_Length` leads SHAP with 27.6% of mean |SHAP|, then `Max` (17.8%) and `Std` (17.3%).
- **Stability under ±5% valid perturbations:** the two methods are similar. Top-5 Jaccard is 0.854
  (benign) and 0.880 (Mirai) for SHAP, against 0.844 and 0.884 for LIME.
- **Repeatability gap** (same unchanged row explained 3 times): SHAP 1.000. LIME 0.648 (benign)
  and 0.564 (Mirai).
- **Recommendation: report SHAP.** TreeExplainer is deterministic, so a SHAP explanation can be
  reproduced exactly. LIME's own random sampling changes its top-5 between runs on the same input,
  so a reported LIME explanation may not be reproducible.
- Under perturbation, the predicted class flipped for 28/500 benign rows and 0/500 Mirai rows.

## 6. Planned for PP2

- **Constrained robustness evaluation** following Vitorino et al. (2023): adversarial examples must
  satisfy validity and coherence, use multiplicative CSI perturbations, and respect the 0.1
  quantisation grid.
- **Replication on official data** (UNB release or `CIC-IoT-2023-canonical-neto`), plus
  recomputing features from the PCAPs.
- **Incident reporting:** generating PDF reports with WeasyPrint.
