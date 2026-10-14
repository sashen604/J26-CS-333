# IT4010 Pillar 4 — Kavish

## Scope
Adversarial robustness, SHAP/LIME explainability,
CUSUM drift detection, incident reporting (WeasyPrint).

## Locked decisions — do not change without asking
- CUSUM defaults: k=0.5σ, h=5σ (Estaji et al., IEEE TII 2025)
- CUSUM is univariate: monitor a named scalar, never a
  raw high-dimensional vector
- Monitored streams: (1) fused windowed anomaly score,
  (2) CSI-branch vs flow-branch disagreement
- EWMA is the required baseline detector
- CSI perturbations are MULTIPLICATIVE, never additive
- Adversarial examples must satisfy validity + coherence
  (Vitorino et al. 2023)
- Fusion robustness is a tested hypothesis, not a claim

## Interface (P3's model is a black box)
predict_proba(X_flow, X_csi) -> probabilities
plus per-branch scores and a fixed feature schema.

## Repo rules
This is a shared four-person repo. Only create or modify
files under p4_robustness/ and tests for them.
Never edit p1_dataset/, p2_sdn/, p3_fusion/.
shared/ is read-only: if the interface looks wrong, tell
me, don't change it.
Never commit anything in data/.

## Rules for you
- One task per session. Ask before touching data files.
- Write a pytest test for every module you add.
- Explain what each file does in plain English when done.
- Never invent results or placeholder numbers.
