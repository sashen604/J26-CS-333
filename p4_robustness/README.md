# Pillar 4: Robustness, explainability and drift detection

Pillar 4 tests how far the dual-layer IoT botnet detector can be trusted. It covers drift detection
(CUSUM, with EWMA as the baseline), explanations (SHAP and LIME), adversarial robustness and incident reporting.
Until P2's AI model is ready, an XGBoost **proxy model** stands in for it, so its accuracy is not a detection result.

## Folder layout

| Path | What is in it |
|---|---|
| `data_prep.py` | Downloads the dataset and loads it (Mirai vs benign, leakage features dropped, 28 features quantised). |
| `proxy_model.py` | Trains, saves and loads the XGBoost proxy behind the `predict_proba` interface. |
| `preflight.py` | Checks that everything the demo needs is in place. |
| `demo.py` | The live PP1 demo, in three parts. |
| `adversarial/` | Empty for now: constrained adversarial evaluation is planned for PP2. |
| `drift/` | CUSUM and EWMA detectors and the monitored-stream definitions. |
| `experiments/` | Scripts that produce the results: drift, explainability and the leakage audit. |
| `reporting/` | Empty for now: WeasyPrint incident reports are planned for PP2. |
| `results/` | Committed results tables, plots and the one-page `SUMMARY.md`. |
| `tests/` | The pytest suite (synthetic data, so no download is needed). |
| `xai/` | SHAP and LIME wrappers and the stability metrics. |
| `CLAUDE.md` | Locked design decisions and repo rules for this pillar. |

## Setup from a fresh clone

Run every command from the **repo root** (the `J26-CS-333` folder). The commands are for Windows
PowerShell. On macOS or Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.
The pinned versions were installed and tested on Python 3.14.

1. Create the virtual environment:
   ```
   python -m venv .venv
   ```
2. Install the pinned requirements:
   ```
   .venv\Scripts\python.exe -m pip install -r p4_robustness/requirements.txt
   ```
3. Download the data (about 67 MB, into `data/ciciot2023_hf/`, which Git ignores):
   ```
   .venv\Scripts\python.exe p4_robustness/data_prep.py
   ```
4. Train and save the proxy model (into `data/models/`, which Git ignores):
   ```
   .venv\Scripts\python.exe p4_robustness/proxy_model.py
   ```

## Running things

**Pre-flight check.** Run this before every demo. It prints PASS or FAIL for the Python
environment, packages, data files, model and results files. Every FAIL comes with the command that
fixes it, and the last line is `READY FOR DEMO` only if everything passed.
```
.venv\Scripts\python.exe p4_robustness/preflight.py
```

**Demo.** Three parts: the leakage audit, drift detection (Scenario C, which also opens its plot)
and SHAP vs LIME. The demo waits for Enter between parts. If one part fails, it prints the error and
moves on to the next.
```
.venv\Scripts\python.exe p4_robustness/demo.py
.venv\Scripts\python.exe p4_robustness/demo.py --quick
```
The full demo reruns the analyses at the committed settings and reproduces the numbers in
`results/`. `--quick` uses fewer calibration streams, seeds and rows so it finishes in well under
3 minutes. Smaller samples give somewhat different numbers, so wherever quick mode differs, it
prints a `[QUICK MODE]` note with the committed figure next to the quick one.

**Tests.**
```
.venv\Scripts\python.exe -m pytest p4_robustness/tests
```

**Regenerating the results** (this overwrites the committed files in `results/`):
```
.venv\Scripts\python.exe p4_robustness/experiments/drift_experiment.py
.venv\Scripts\python.exe p4_robustness/experiments/xai_experiment.py
.venv\Scripts\python.exe p4_robustness/experiments/leakage_audit.py
```
The leakage audit writes no files. It checks its figures against `data/README.md` and exits with an
error if any differ.

## Results and their limitations

- `results/SUMMARY.md`: a one-page plain-English summary of Pillar 4 so far.
- `results/drift_results.md`: CUSUM vs EWMA, calibration and scenarios A, B and C, with caveats.
- `results/xai_results.md`: the global SHAP ranking, SHAP vs LIME stability and repeatability.
- `data/README.md`: where the data comes from, the window-size leakage evidence and fix, and the
  limitations. The main limitations: the data is an unofficial mirror whose features are **not**
  official CICIoT2023, it covers Mirai vs benign only, and replication on the official data is
  still pending.

Results must be labelled "CICIoT2023 subset (HF mirror)", not "CICIoT2023".
