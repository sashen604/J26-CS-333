"""Live demo of Pillar 4 for the Progress Presentation (PP1).

Run from the repo root:  python p4_robustness/demo.py [--quick]

Three parts, run in order. After each part the demo waits for Enter so the presenter can talk.
  Part 1: window-size leakage audit (experiments/leakage_audit.py).
  Part 2: drift, Scenario C, CUSUM vs EWMA calibrated to an equal false-alarm rate;
          opens results/drift_scenario_c.png.
  Part 3: XAI, top-5 features by mean |SHAP| and the SHAP vs LIME repeatability gap.

--quick uses fewer calibration streams, seeds and rows so the compute takes under 3 minutes.
Whenever a quick-mode setting differs from the committed full results, a note says so and
shows the committed figures from results/drift_results.md and results/xai_results.md.
If a part fails, its error is printed and the demo moves on to the next part.

Data: "CICIoT2023 subset (HF mirror)"; see data/README.md. Model: XGBoost proxy, not P2's
AI model.
"""
import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

P4_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(P4_DIR))
sys.path.insert(0, str(P4_DIR / "experiments"))

import data_prep  # noqa: E402

RESULTS_DIR = P4_DIR / "results"
TOP_N = 5


@dataclass
class Settings:
    calib_streams: int   # benign streams used to calibrate h and L
    n_seeds: int         # Scenario C seeds
    n_summary: int       # rows for the global SHAP ranking
    n_repeat_rows: int   # rows per class for E-XAI repeatability


# FULL matches the committed experiments (drift_experiment.py, xai_experiment.py).
FULL = Settings(calib_streams=500, n_seeds=20, n_summary=1000, n_repeat_rows=50)
QUICK = Settings(calib_streams=100, n_seeds=5, n_summary=200, n_repeat_rows=10)


def heading(title, explanation):
    bar = "=" * 78
    print(f"\n{bar}\n  {title.upper()}\n{bar}")
    print(f"What you are seeing: {explanation}\n")


def quick_note(text):
    print(f"  [QUICK MODE] {text}")


def wait_for_enter(prompt="\n-- Press Enter to continue --"):
    try:
        input(prompt)
    except EOFError:
        print()


def open_file(path):
    """Open a file in the system viewer; never fails the demo."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        print(f"Opened {path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not open {path}: {exc}")


# ---------------------------------------------------------------------------
# Committed figures, read from the results files (never typed in by hand)
# ---------------------------------------------------------------------------
def committed_scenario_c(path=RESULTS_DIR / "drift_results.md"):
    """{detector: (param text, delay text, false alarms text, missed text)} from the Scenario C table."""
    text = path.read_text(encoding="utf-8")
    section = text.split("## Scenario C", 1)[1]
    rows = re.findall(r"^\| (CUSUM|EWMA) \| ([hL] = [\d.]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$",
                      section, flags=re.M)
    return {d: (p, delay.strip(), fa.strip(), missed.strip()) for d, p, delay, fa, missed in rows}


def committed_calibration(path=RESULTS_DIR / "drift_results.md"):
    """{detector: (param text, false alarms per 1000)} for the calibrated rows."""
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (CUSUM|EWMA) \| calibrated \| ([hL] = [\d.]+) \| ([\d.]+) \|$", text, flags=re.M)
    return {d: (p, far) for d, p, far in rows}


def committed_shap_top(path=RESULTS_DIR / "xai_results.md", n=TOP_N):
    """[(feature, mean |SHAP|, share)] for the top n rows of the committed global SHAP table."""
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| ([^|]+) \| ([\d.]+) \| ([\d.]+) \|$", text, flags=re.M)
    return [(f.strip(), float(v), float(s)) for _, f, v, s in rows[:n]]


def committed_repeatability(path=RESULTS_DIR / "xai_results.md"):
    """{(method, class): 'mean ± std'} from the committed E-XAI repeatability table."""
    text = path.read_text(encoding="utf-8")
    section = text.split("## E-XAI repeatability", 1)[1]
    rows = re.findall(r"^\| (SHAP|LIME) \| (benign|Mirai) \| ([\d.]+ ± [\d.]+) \|$", section, flags=re.M)
    return {(m, c): v for m, c, v in rows}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def part1_leakage(settings, quick):
    heading("Part 1: window-size leakage audit",
            "the raw data leaks the capture window size (10 packets benign, 100 Mirai), so a "
            "trivial rule can 'detect' Mirai; dropping and quantising features removes it.")
    import leakage_audit
    r = leakage_audit.run_audit()
    leakage_audit.print_report(r)
    diffs = leakage_audit.compare_with_readme(r)
    if diffs:
        print("\nMISMATCH with data/README.md:")
        for k, got, want in diffs:
            print(f"    {k}: measured {got}, README {want}")
    else:
        print("\nAll figures match data/README.md.")
    if quick:
        quick_note("Part 1 is identical in quick mode (full test split).")


def part2_drift(settings, quick):
    heading("Part 2: drift detection, Scenario C (CUSUM vs EWMA)",
            "benign traffic slowly changes (5 features x1.0 -> x2.0 from window 200); both detectors "
            "are tuned to the same false-alarm rate, then we compare how fast each notices.")
    import drift_experiment as de
    from proxy_model import ProxyFusionModel

    val_benign, test_benign, test_mirai = de.load_scores()
    ref = de.fit_reference(de.window_logits(val_benign))
    print(f"Reference: {ref.n_windows} benign validation windows, mu0 = {ref.mu0:.4f}, sigma = {ref.sigma:.4f}")

    calib_seeds = range(de.CALIB_SEEDS.start, de.CALIB_SEEDS.start + settings.calib_streams)
    calib = [de.build_stream(test_benign, test_mirai, np.zeros(de.CALIB_WINDOWS, int), s) for s in calib_seeds]
    h, far_c = de.calibrate("cusum", ref, calib)
    L, far_e = de.calibrate("ewma", ref, calib)
    print(f"Calibrated on {len(calib)} benign streams ({len(calib) * de.CALIB_WINDOWS:,} windows) "
          f"to {de.TARGET_FAR * 1000:g} false alarm per 1000 windows:")
    print(f"    CUSUM h = {h:.3f}  ({far_c * 1000:.3f} per 1000)")
    print(f"    EWMA  L = {L:.3f}  ({far_e * 1000:.3f} per 1000)")
    committed_cal = committed_calibration()
    if quick:
        quick_note(f"calibrated on {settings.calib_streams} streams; the committed results use "
                   f"{FULL.calib_streams} (CUSUM {committed_cal['CUSUM'][0]}, EWMA {committed_cal['EWMA'][0]}).")

    model = ProxyFusionModel.load()
    test = data_prep.load_split("test")
    benign_df = test[test[data_prep.TARGET] == 0].reset_index(drop=True)
    scaled = de.select_scale_safe(de.gain_shares(model))
    print(f"Scaled features: {', '.join(scaled)}")
    streams = []
    for s in range(settings.n_seeds):
        frame, _ = de.build_scenario_c(benign_df, scaled, s)
        streams.append(de.window_logits(model.predict_proba(frame)[:, 1]))

    print(f"\nScenario C over {settings.n_seeds} seeds (delay in windows after drift starts at window 200):")
    print(f"    {'Detector':9} {'Parameter':12} {'Delay':>16} {'False alarms <200':>19} {'Missed':>7}")
    for kind, label, param in (("cusum", "CUSUM", f"h = {h:.3f}"), ("ewma", "EWMA", f"L = {L:.3f}")):
        r = de.run_streams(kind, ref, h if kind == "cusum" else L, streams)
        print(f"    {label:9} {param:12} {de.fmt(r['delay_mean'], r['delay_std']):>16} "
              f"{de.fmt(r['fa_mean'], r['fa_std']):>19} {str(r['missed']) + '/' + str(r['n']):>7}")
    committed = committed_scenario_c()
    if quick:
        quick_note(f"{settings.n_seeds} seeds instead of {FULL.n_seeds}. Committed full results "
                   "(drift_results.md):")
        for d, (p, delay, fa, missed) in committed.items():
            print(f"      {d:6} {p:10} delay {delay}, false alarms {fa}, missed {missed}")

    print("\nOpening the committed plot (Scenario C, seed 0, full calibration).")
    open_file(RESULTS_DIR / "drift_scenario_c.png")


def part3_xai(settings, quick):
    heading("Part 3: explainability, SHAP vs LIME",
            "which features drive the model's decisions, and whether asking the same explainer "
            "twice about the same row gives the same answer.")
    from proxy_model import ProxyFusionModel
    from xai.explainers import lime_explain, make_lime_explainer, make_shap_explainer, shap_explain
    from xai.metrics import exai_repeatability
    import xai_experiment as xe

    model = ProxyFusionModel.load()
    test = data_prep.load_split("test")
    train = data_prep.load_split("train")
    features = model.feature_names

    rows = xe.sample_rows(test, settings.n_summary, xe.SEED)
    sv = shap_explain(model, rows, make_shap_explainer(model))
    ranking = xe.mean_abs_shap(sv.values, features)
    share = ranking / ranking.sum()
    print(f"Top {TOP_N} features by mean |SHAP| on {settings.n_summary:,} test rows:")
    for i, f in enumerate(ranking.index[:TOP_N], 1):
        print(f"    {i}. {f:15} mean |SHAP| {ranking[f]:.4f}  share {share[f]:.3f}")
    if quick:
        quick_note(f"{settings.n_summary} rows instead of {FULL.n_summary}. Committed top {TOP_N} "
                   "(xai_results.md):")
        for i, (f, v, s) in enumerate(committed_shap_top(), 1):
            print(f"      {i}. {f:15} mean |SHAP| {v:.4f}  share {s:.3f}")

    lime_ref = xe.sample_rows(train, xe.N_LIME_REFERENCE, xe.SEED)
    shap_explainer = make_shap_explainer(model)
    lime_free = make_lime_explainer(model, lime_ref, xe.SEED)  # seeded once, not per call

    def shap_fn(r):
        return shap_explain(model, r, shap_explainer).values[0]

    def lime_fn(r):
        return lime_explain(model, r, lime_ref, k=xe.K, explainer=lime_free,
                            num_samples=xe.LIME_SAMPLES, seed=None).weights

    n = settings.n_repeat_rows
    print(f"\nE-XAI repeatability: explain the same unchanged row {xe.N_REPEAT} times; share of the top-{xe.K} "
          f"kept every time (1.0 = identical). {n} rows per class:")
    for cls_name, label in ((0, "benign"), (1, "Mirai")):
        cls_rows = xe.sample_rows(test[test[data_prep.TARGET] == cls_name], n, xe.SEED)
        for method, fn in (("SHAP", shap_fn), ("LIME", lime_fn)):
            rep = [exai_repeatability(fn, cls_rows.iloc[[i]], xe.N_REPEAT, xe.K) for i in range(n)]
            print(f"    {method} {label:6}: {np.mean(rep):.3f} ± {np.std(rep, ddof=1):.3f}")
    if quick:
        quick_note(f"{n} rows per class instead of {FULL.n_repeat_rows}. Committed (xai_results.md):")
        for (m, c), v in committed_repeatability().items():
            print(f"      {m} {c:6}: {v}")
    print("\nSHAP TreeExplainer is deterministic, so its explanation can be reproduced exactly; "
          "LIME's random sampling changes its top-5 between runs on the same row.")


PARTS = (part1_leakage, part2_drift, part3_xai)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="fewer streams, seeds and rows (compute under 3 minutes)")
    args = parser.parse_args(argv)
    if not sys.stdout.isatty():  # piped or redirected output: keep "±" readable
        sys.stdout.reconfigure(encoding="utf-8")
    settings = QUICK if args.quick else FULL
    print("Pillar 4 demo: CICIoT2023 subset (HF mirror), Mirai vs benign; XGBoost proxy, not P2's AI model.")
    if args.quick:
        print("Running in QUICK mode: notes mark every place where it differs from the committed full results.")
    compute = 0.0
    failed = []
    for i, part in enumerate(PARTS):
        start = time.perf_counter()
        try:
            part(settings, args.quick)
        except Exception:  # noqa: BLE001 — one failed part must not end the demo
            failed.append(part.__name__)
            print(f"\n!! {part.__name__} failed; continuing with the next part.")
            traceback.print_exc(file=sys.stdout)
        elapsed = time.perf_counter() - start
        compute += elapsed
        print(f"\n(part took {elapsed:.1f} s)")
        if i < len(PARTS) - 1:
            wait_for_enter()
    print(f"\nDemo finished. Compute time {compute:.1f} s." + (f" Failed parts: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
