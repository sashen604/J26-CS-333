"""Drift detection experiment for Pillar 4: CUSUM vs the EWMA baseline.

Run from the repo root:  python p4_robustness/experiments/drift_experiment.py

What it does
------------
1. Monitored scalar: the LOGIT of stream (1), the windowed P(attack) from the
   proxy model's predict_proba (window = 50 rows), with p clipped to
   [1e-6, 1 - 1e-6].
2. Reference (mu0, sigma): benign-only VALIDATION rows, non-overlapping
   windows. A sigma floor (default 1e-3) is applied only if the measured
   sigma is below it; whether it was used is recorded.
3. Streams are built from the TEST split only. The data has no time order,
   so each stream samples rows without replacement using a fixed seed:
     Scenario A (abrupt):  200 benign windows, then 100 all-Mirai windows.
     Scenario B (gradual): 200 benign windows, then 100 windows whose Mirai
                           share rises linearly from 0% to 30%.
   Drift starts at window 200.
4. Calibration: CUSUM's h and EWMA's L are tuned by bisection so each gives
   about 1 false alarm per 1000 windows on benign-only test streams of 300
   windows. k = 0.5 and lam = 0.2 stay fixed. Results are also reported at the
   uncalibrated defaults h = 5 and L = 3.
5. 20 seeds per scenario and detector: detection delay, false alarms before
   window 200, and missed detections.

Data: "CICIoT2023 subset (HF mirror)", Mirai vs benign; see data/README.md.
The proxy model's near-perfect separation on this subset makes the
attack-side shift very large, so detection here is expected to be easy.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

P4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4_DIR))

import data_prep  # noqa: E402
from drift.cusum import CUSUM  # noqa: E402
from drift.ewma import EWMA  # noqa: E402
from drift.streams import FUSED_ANOMALY_SCORE  # noqa: E402

RESULTS_DIR = P4_DIR / "results"
MONITORED = f"logit_{FUSED_ANOMALY_SCORE}"

WINDOW = 50
N_BEFORE = 200
N_AFTER = 100
RAMP_MAX_SHARE = 0.30
CLIP_EPS = 1e-6
SIGMA_FLOOR = 1e-3
TARGET_FAR = 1e-3  # false alarms per window
N_SEEDS = 20
EVAL_SEEDS = range(N_SEEDS)
CALIB_SEEDS = range(1000, 1500)  # 500 benign streams, disjoint from evaluation seeds
CALIB_WINDOWS = 300
DEFAULT_H = 5.0
DEFAULT_L = 3.0


def logit(p, eps=CLIP_EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def window_logits(row_scores, window=WINDOW):
    """Logit of the mean P(attack) over consecutive non-overlapping windows."""
    row_scores = np.asarray(row_scores, dtype=float)
    n = len(row_scores) // window
    return logit(row_scores[: n * window].reshape(n, window).mean(axis=1))


def ramp_mirai_counts(n_windows=N_AFTER, max_share=RAMP_MAX_SHARE, window=WINDOW):
    """Mirai rows per window, rising linearly from 0% to max_share over n_windows."""
    shares = np.linspace(0.0, max_share, n_windows)
    return np.rint(shares * window).astype(int)


def mirai_counts(scenario, n_before=N_BEFORE, n_after=N_AFTER, window=WINDOW):
    """Mirai rows in each window of a scenario: 'A' abrupt, 'B' gradual, 'benign' none."""
    before = np.zeros(n_before, dtype=int)
    if scenario == "A":
        after = np.full(n_after, window)
    elif scenario == "B":
        after = ramp_mirai_counts(n_after, window=window)
    elif scenario == "benign":
        after = np.zeros(n_after, dtype=int)
    else:
        raise ValueError("scenario must be 'A', 'B' or 'benign'")
    return np.concatenate([before, after])


def build_stream(benign_scores, mirai_scores, counts, seed, window=WINDOW):
    """Sample rows (without replacement) into windows with the given Mirai counts per window.

    Returns the per-window logit stream. Rows' P(attack) are precomputed by the
    model, so sampling scores is the same as sampling rows and calling predict_proba.
    """
    counts = np.asarray(counts, dtype=int)
    n_mirai = int(counts.sum())
    n_benign = int(len(counts) * window - n_mirai)
    if n_benign > len(benign_scores) or n_mirai > len(mirai_scores):
        raise ValueError("Not enough rows to sample this stream without replacement")
    rng = np.random.default_rng(seed)
    b = rng.choice(benign_scores, n_benign, replace=False)
    m = rng.choice(mirai_scores, n_mirai, replace=False)
    windows, bi, mi = [], 0, 0
    for c in counts:
        windows.append(np.concatenate([b[bi: bi + window - c], m[mi: mi + c]]))
        bi += window - c
        mi += c
    return window_logits(np.concatenate(windows), window)


@dataclass
class Reference:
    mu0: float
    sigma: float
    measured_sigma: float
    floor_used: bool
    n_windows: int


def fit_reference(values, sigma_floor=SIGMA_FLOOR):
    """mu0 and sigma (ddof=1) of the reference windows, with the sigma floor applied if needed."""
    values = np.asarray(values, dtype=float)
    measured = float(values.std(ddof=1))
    floor_used = measured < sigma_floor
    return Reference(float(values.mean()), sigma_floor if floor_used else measured,
                     measured, floor_used, len(values))


def make_detector(kind, ref, param):
    """kind 'cusum' -> param is h; kind 'ewma' -> param is L. Uses the shared reference."""
    det = CUSUM(MONITORED, h=param) if kind == "cusum" else EWMA(MONITORED, L=param)
    det.fit_reference([ref.mu0 - 1.0, ref.mu0 + 1.0])  # sets up state; mu0/sigma replaced below
    det.mu0, det.sigma = ref.mu0, ref.sigma
    det.reset()
    return det


def alarm_indices(kind, ref, param, stream):
    det = make_detector(kind, ref, param)
    return [i for i, x in enumerate(stream) if det.update(x)]


def false_alarm_rate(kind, ref, param, streams):
    alarms = sum(len(alarm_indices(kind, ref, param, s)) for s in streams)
    return alarms / sum(len(s) for s in streams)


def calibrate(kind, ref, streams, target=TARGET_FAR, lo=0.05, hi=60.0, iters=30):
    """Bisection on h (CUSUM) or L (EWMA) so the false alarm rate is closest to target."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        far = false_alarm_rate(kind, ref, mid, streams)
        if best is None or abs(far - target) < abs(best[1] - target) or \
                (abs(far - target) == abs(best[1] - target) and mid < best[0]):
            best = (mid, far)
        if far > target:
            lo = mid
        else:
            hi = mid
    return best


def score_run(alarms, drift_start=N_BEFORE):
    """Delay (None if missed), false alarms before drift, and whether it was missed."""
    false_alarms = sum(1 for a in alarms if a < drift_start)
    after = [a for a in alarms if a >= drift_start]
    delay = after[0] - drift_start if after else None
    return delay, false_alarms, delay is None


def run_scenario(kind, ref, param, benign_scores, mirai_scores, scenario, seeds=EVAL_SEEDS):
    counts = mirai_counts(scenario)
    rows = [score_run(alarm_indices(kind, ref, param,
                                    build_stream(benign_scores, mirai_scores, counts, s)))
            for s in seeds]
    delays = [d for d, _, _ in rows if d is not None]
    fas = [f for _, f, _ in rows]
    missed = sum(m for _, _, m in rows)
    return {
        "delay_mean": float(np.mean(delays)) if delays else None,
        "delay_std": float(np.std(delays, ddof=1)) if len(delays) > 1 else None,
        "fa_mean": float(np.mean(fas)),
        "fa_std": float(np.std(fas, ddof=1)),
        "missed": int(missed),
        "n": len(rows),
    }


def load_scores():
    """Per-row P(attack) from the saved proxy model for validation-benign, test-benign, test-Mirai."""
    from proxy_model import ProxyFusionModel
    model = ProxyFusionModel.load()
    val = data_prep.load_split("validation")
    test = data_prep.load_split("test")
    val_benign = model.predict_proba(val[val[data_prep.TARGET] == 0])[:, 1]
    test_benign = model.predict_proba(test[test[data_prep.TARGET] == 0])[:, 1]
    test_mirai = model.predict_proba(test[test[data_prep.TARGET] == 1])[:, 1]
    return val_benign, test_benign, test_mirai


def fmt(mean, std, digits=2):
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if std is not None else f"{mean:.{digits}f}"


def plot_scenario_b(stream, ref, h, L, path, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = alarm_indices("cusum", ref, h, stream)
    e = alarm_indices("ewma", ref, L, stream)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(stream, color="0.35", lw=1, label=MONITORED)
    ax.axvline(N_BEFORE, color="tab:red", ls="--", lw=1.2, label="drift start (window 200)")
    ax.axhline(ref.mu0, color="0.6", ls=":", lw=1, label="reference mu0")
    ax.scatter(c, stream[c], marker="v", s=60, color="tab:blue", zorder=3,
               label=f"CUSUM alarms (h={h:.2f})")
    ax.scatter(e, stream[e], marker="^", s=60, color="tab:orange", zorder=3,
               label=f"EWMA alarms (L={L:.2f})")
    ax.set_xlabel("window (50 rows each)")
    ax.set_ylabel("logit of windowed P(attack)")
    ax.set_title(f"Scenario B (gradual, Mirai share 0% to 30%), seed {seed}, calibrated detectors\n"
                 "CICIoT2023 subset (HF mirror), proxy model")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val_benign, test_benign, test_mirai = load_scores()

    ref_windows = window_logits(val_benign)
    ref = fit_reference(ref_windows)
    raw_windows = val_benign[: len(val_benign) // WINDOW * WINDOW].reshape(-1, WINDOW).mean(axis=1)
    print(f"Reference: {ref.n_windows} benign validation windows; logit mu0={ref.mu0:.4f}, "
          f"sigma={ref.sigma:.4f} (measured {ref.measured_sigma:.4f}, floor used: {ref.floor_used})")

    calib_streams = [build_stream(test_benign, test_mirai, np.zeros(CALIB_WINDOWS, int), s)
                     for s in CALIB_SEEDS]
    n_calib = len(calib_streams) * CALIB_WINDOWS
    h_cal, far_c = calibrate("cusum", ref, calib_streams)
    L_cal, far_e = calibrate("ewma", ref, calib_streams)
    far_c_def = false_alarm_rate("cusum", ref, DEFAULT_H, calib_streams)
    far_e_def = false_alarm_rate("ewma", ref, DEFAULT_L, calib_streams)
    print(f"Calibrated: CUSUM h={h_cal:.3f} (FAR {far_c * 1000:.3f}/1000), "
          f"EWMA L={L_cal:.3f} (FAR {far_e * 1000:.3f}/1000)")
    print(f"Uncalibrated: CUSUM h=5 FAR {far_c_def * 1000:.3f}/1000, EWMA L=3 FAR {far_e_def * 1000:.3f}/1000")

    settings = [("calibrated", h_cal, L_cal), ("uncalibrated", DEFAULT_H, DEFAULT_L)]
    results = {}
    for label, h, L in settings:
        for scen in ("A", "B"):
            for kind, param in (("cusum", h), ("ewma", L)):
                r = run_scenario(kind, ref, param, test_benign, test_mirai, scen)
                results[(label, scen, kind)] = (param, r)
                print(f"[{label}] scenario {scen} {kind:5} param={param:.3f}: delay {fmt(r['delay_mean'], r['delay_std'])}, "
                      f"false alarms before 200 {fmt(r['fa_mean'], r['fa_std'])}, missed {r['missed']}/{r['n']}")

    plot_seed = EVAL_SEEDS[0]
    stream_b = build_stream(test_benign, test_mirai, mirai_counts("B"), plot_seed)
    plot_scenario_b(stream_b, ref, h_cal, L_cal, RESULTS_DIR / "drift_scenario_b.png", plot_seed)

    lines = [
        "# Drift detection results: CUSUM vs EWMA (Pillar 4)",
        "",
        "Generated by `python p4_robustness/experiments/drift_experiment.py`. Do not edit by hand.",
        "",
        "**Data:** CICIoT2023 subset (HF mirror), Mirai vs benign, `Number` excluded. See `data/README.md` "
        "for provenance and limitations. **Model:** XGBoost proxy, which is not P3's fusion model.",
        "",
        "**Caveat:** benign and Mirai are almost perfectly separable on this subset, so the attack-side shift "
        "in the monitored scalar is very large. Fast detection here reflects that, not the detectors' "
        "sensitivity to subtle drift.",
        "",
        f"**Caveat:** all {len(CALIB_SEEDS) + N_SEEDS} streams ({len(CALIB_SEEDS)} for calibration, {N_SEEDS} per "
        "scenario) resample the same test rows, so the ± values likely understate the true uncertainty.",
        "",
        "## Setup",
        "",
        f"- Monitored scalar: `{MONITORED}`, the logit of the mean P(attack) over {WINDOW}-row windows, "
        f"with p clipped to [{CLIP_EPS:g}, 1 - {CLIP_EPS:g}].",
        f"- Reference: {ref.n_windows} benign-only validation windows. mu0 = {ref.mu0:.4f}, "
        f"sigma = {ref.sigma:.4f} (measured {ref.measured_sigma:.4f}). Sigma floor {SIGMA_FLOOR:g} used: "
        f"**{'yes' if ref.floor_used else 'no'}**.",
        f"- For comparison, the raw windowed P(attack) on the same windows: mean = {raw_windows.mean():.3e}, "
        f"sigma = {raw_windows.std(ddof=1):.3e}.",
        f"- Streams come from the test split, sampled without replacement with a fixed seed per run. "
        f"{N_BEFORE} benign windows, then {N_AFTER} drift windows. Drift starts at window {N_BEFORE}.",
        "  - Scenario A (abrupt): every drift window is 100% Mirai.",
        f"  - Scenario B (gradual): Mirai share rises linearly from 0% to {RAMP_MAX_SHARE:.0%} "
        f"(rounded to whole rows).",
        "- CUSUM k = 0.5 sigma. EWMA lambda = 0.2. Both restart automatically after an alarm.",
        f"- Seeds {EVAL_SEEDS.start}–{EVAL_SEEDS.stop - 1} for the scenarios. Values below are mean ± std "
        f"over {N_SEEDS} seeds (sample std).",
        "",
        "## Calibration (target: 1 false alarm per 1000 windows)",
        "",
        f"Benign-only test streams: {len(calib_streams)} streams × {CALIB_WINDOWS} windows = {n_calib:,} windows "
        f"(seeds {CALIB_SEEDS.start}–{CALIB_SEEDS.stop - 1}). One 300-window stream cannot measure a rate of "
        "1/1000, so many streams were pooled. All streams draw from the same 19,999 benign test rows, "
        "so they are not independent.",
        "",
        "| Detector | Setting | Parameter | False alarms per 1000 windows |",
        "|---|---|---|---|",
        f"| CUSUM | calibrated | h = {h_cal:.3f} | {far_c * 1000:.3f} |",
        f"| EWMA | calibrated | L = {L_cal:.3f} | {far_e * 1000:.3f} |",
        f"| CUSUM | uncalibrated | h = {DEFAULT_H:g} | {far_c_def * 1000:.3f} |",
        f"| EWMA | uncalibrated | L = {DEFAULT_L:g} | {far_e_def * 1000:.3f} |",
        "",
    ]
    for label, _, _ in settings:
        lines += [
            f"## Results: {label}",
            "",
            "| Scenario | Detector | Parameter | Detection delay (windows) | False alarms before window 200 | Missed |",
            "|---|---|---|---|---|---|",
        ]
        for scen, name in (("A", "A abrupt"), ("B", "B gradual")):
            for kind, pname in (("cusum", "h"), ("ewma", "L")):
                param, r = results[(label, scen, kind)]
                lines.append(f"| {name} | {kind.upper()} | {pname} = {param:.3f} | "
                             f"{fmt(r['delay_mean'], r['delay_std'])} | {fmt(r['fa_mean'], r['fa_std'])} | "
                             f"{r['missed']}/{r['n']} |")
        lines.append("")
    first_mirai = int(np.argmax(ramp_mirai_counts() > 0))
    lines += [
        "Detection delay is the window index of the first alarm at or after window 200, minus 200, so 0 means "
        "an alarm in the first drift window. It is averaged over seeds that detected the drift.",
        "",
        f"**Scenario B note:** after rounding to whole rows, the first {first_mirai} ramp windows contain 0 Mirai "
        f"rows. Window {N_BEFORE + first_mirai} is the first to contain any (1 of {WINDOW}). A delay of "
        f"{first_mirai} therefore means an alarm at the first window containing any Mirai traffic. On this "
        "subset, one Mirai row in a window is enough to push the logit far above the reference.",
        "",
        f"Plot: `drift_scenario_b.png` (Scenario B, seed {plot_seed}, calibrated detectors).",
        "",
    ]
    (RESULTS_DIR / "drift_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'drift_results.md'} and {RESULTS_DIR / 'drift_scenario_b.png'}")


if __name__ == "__main__":
    main()
