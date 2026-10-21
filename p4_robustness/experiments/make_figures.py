"""Presentation figures for PP1, drawn from the COMMITTED results (and, for figure 2, the raw test split).

Run from the repo root:  python p4_robustness/experiments/make_figures.py

No experiment is rerun. Figures 1 and 3-6 parse every number from data/README.md,
results/drift_results.md or results/xai_results.md. Figure 2 counts Number values in the
raw test split and raises unless its two headline shares match data/README.md. If a figure
cannot be found, the script raises instead of guessing. Output: results/figures/*.png,
16:9, white background, large fonts for slides.

1. leakage.png                 "off the 0.1 grid -> Mirai" accuracy before vs after quantising
2. leakage_number.png          Number value distribution by class (10, 100, other), raw test split
3. drift_calibration.png       false alarms per 1000 windows, default vs calibrated
4. drift_scenario_c_delay.png  Scenario C detection delay, mean ± std over seeds
5. shap_top10.png              top 10 features by mean |SHAP|
6. repeatability.png           SHAP vs LIME E-XAI repeatability by class
"""
import re
import sys
from pathlib import Path

P4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4_DIR))

import demo  # noqa: E402  (its committed-results parsers)

REPO_ROOT = P4_DIR.parent
DATA_README = REPO_ROOT / "data" / "README.md"
RESULTS_DIR = P4_DIR / "results"
DRIFT_MD = RESULTS_DIR / "drift_results.md"
XAI_MD = RESULTS_DIR / "xai_results.md"
FIG_DIR = RESULTS_DIR / "figures"

CAPTION = "HF mirror features, not official CICIoT2023"
SIZE = (13.333, 7.5)  # inches, 16:9
DPI = 150

# Colours: the dataviz reference palette's first two categorical slots, validated on white.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


# ---------------------------------------------------------------------------
# Parsing the committed files
# ---------------------------------------------------------------------------
def _search(pattern, text, what):
    m = re.search(pattern, text, flags=re.S)
    if not m:
        raise ValueError(f"Could not find {what} in the committed files")
    return m


def leakage_rule(path=DATA_README):
    """(accuracy before, accuracy after) of the off-grid rule; after equals the majority baseline."""
    text = path.read_text(encoding="utf-8")
    before = _search(r'off the 0\.1 grid → Mirai" scored \*\*([\d.]+)\*\*', text, "the before-rounding accuracy")
    after = _search(r"the same rule scores \*\*([\d.]+)\*\*,\s+exactly the majority-class baseline", text,
                    "the after-rounding accuracy and majority baseline")
    return float(before.group(1)), float(after.group(1))


def number_shares(path=DATA_README):
    """{'benign': (pct, rows, total), 'Mirai': (...)} for Number = 10 and Number = 100."""
    text = path.read_text(encoding="utf-8")
    m = _search(r"is 10 for ([\d.]+)% of benign rows \(([\d,]+) of ([\d,]+)\) and 100 for ([\d.]+)% of Mirai\s+"
                r"rows \(([\d,]+) of ([\d,]+)\)", text, "the Number shares")
    g = [m.group(i).replace(",", "") for i in range(1, 7)]
    return {"benign": (float(g[0]), int(g[1]), int(g[2])), "Mirai": (float(g[3]), int(g[4]), int(g[5]))}


def calibration(path=DRIFT_MD):
    """{(detector, setting): (parameter text, false alarms per 1000)} from the calibration table."""
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (CUSUM|EWMA) \| (calibrated|uncalibrated) \| ([hL] = [\d.]+) \| ([\d.]+) \|$",
                      text, flags=re.M)
    out = {(d, s): (p, float(v)) for d, s, p, v in rows}
    if len(out) != 4:
        raise ValueError(f"Expected 4 calibration rows in {path.name}, found {len(out)}")
    return out


def target_far(path=DRIFT_MD):
    text = path.read_text(encoding="utf-8")
    m = _search(r"## Calibration \(target: ([\d.]+) false alarm per 1000 windows\)", text, "the calibration target")
    return float(m.group(1))


def mean_std(text):
    m = re.fullmatch(r"([\d.]+) ± ([\d.]+)", text.strip())
    if not m:
        raise ValueError(f"Not a 'mean ± std' value: {text!r}")
    return float(m.group(1)), float(m.group(2))


def scenario_c_delays(path=DRIFT_MD):
    """{detector: (parameter text, mean, std, missed text)} from the Scenario C table."""
    rows = demo.committed_scenario_c(path)
    if set(rows) != {"CUSUM", "EWMA"}:
        raise ValueError(f"Scenario C table in {path.name} is incomplete")
    return {d: (p, *mean_std(delay), missed) for d, (p, delay, _, missed) in rows.items()}


def n_seeds(path=DRIFT_MD):
    m = _search(r"mean ± std over (\d+) seeds", path.read_text(encoding="utf-8"), "the number of seeds")
    return int(m.group(1))


def shap_top10(path=XAI_MD):
    rows = demo.committed_shap_top(path, n=10)
    if len(rows) != 10:
        raise ValueError(f"Expected 10 SHAP rows in {path.name}, found {len(rows)}")
    return rows


def shap_rows(path=XAI_MD):
    m = _search(r"([\d,]+) random test rows", path.read_text(encoding="utf-8"), "the SHAP row count")
    return int(m.group(1).replace(",", ""))


def repeatability(path=XAI_MD):
    """{(method, class): (mean, std)}."""
    rows = {k: mean_std(v) for k, v in demo.committed_repeatability(path).items()}
    if len(rows) != 4:
        raise ValueError(f"Expected 4 repeatability rows in {path.name}, found {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 20, "axes.titlesize": 26, "axes.labelsize": 20,
        "xtick.labelsize": 20, "ytick.labelsize": 18, "legend.fontsize": 18,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _new(plt, title):
    fig, ax = plt.subplots(figsize=SIZE)
    ax.set_title(title, loc="left", pad=20, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, name, out_dir):
    fig.text(0.01, 0.015, CAPTION, fontsize=14, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = Path(out_dir) / name
    fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _inside_labels(ax, bars, texts):
    """White labels just inside the top of each bar, clear of lines drawn above the bars."""
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.96, t, ha="center", va="top",
                fontsize=19, color="white", fontweight="bold")


def _bar_labels(ax, bars, texts, pad=0.01):
    top = ax.get_ylim()[1]
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad * top, t,
                ha="center", va="bottom", fontsize=20, color=INK)


def fig_leakage(out_dir):
    plt = _plt()
    before, after = leakage_rule()
    fig, ax = _new(plt, 'Quantising kills the "off the 0.1 grid → Mirai" rule')
    bars = ax.bar(["Before quantising", "After quantising"], [before, after], width=0.5, color=BLUE)
    ax.axhline(after, color=INK_2, ls="--", lw=2)
    ax.text(1.32, after, f"majority-class\nbaseline {after:.4f}", va="center", ha="left", fontsize=17, color=INK_2)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    _bar_labels(ax, bars, [f"{before:.4f}", f"{after:.4f}"])
    return _save(fig, "leakage.png", out_dir)


NUMBER_GROUPS = ("10", "100", "Other")


def number_groups(df):
    """{class: {group: rows}} for Number = 10, Number = 100 and every other value, plus class totals."""
    import data_prep
    out = {}
    for c, name in ((0, "benign"), (1, "Mirai")):
        n = df.loc[df[data_prep.TARGET] == c, "Number"]
        ten, hundred = int((n == 10).sum()), int((n == 100).sum())
        out[name] = {"10": ten, "100": hundred, "Other": len(n) - ten - hundred, "total": len(n)}
    return out


def check_headline(groups, documented):
    """Raise unless benign at 10 and Mirai at 100 match data/README.md (share to 2 dp, and counts)."""
    for name, group in (("benign", "10"), ("Mirai", "100")):
        rows, total = groups[name][group], groups[name]["total"]
        measured = (round(100 * rows / total, 2), rows, total)
        if measured != documented[name]:
            raise ValueError(f"{name} Number = {group}: raw test split gives {measured}, "
                             f"data/README.md says {documented[name]}")


def raw_number_groups():
    """Number groups counted from the raw (unquantised) test split, checked against data/README.md."""
    import leakage_audit
    groups = number_groups(leakage_audit.load_raw("test"))
    check_headline(groups, number_shares())
    return groups


def fig_leakage_number(out_dir):
    import numpy as np
    plt = _plt()
    g = raw_number_groups()
    fig, ax = _new(plt, "Benign windows hold 10 packets, Mirai windows 100")
    x = np.arange(len(NUMBER_GROUPS))
    w = 0.38
    headline = {"benign": "10", "Mirai": "100"}
    for i, (name, color) in enumerate((("benign", BLUE), ("Mirai", ORANGE))):
        counts = [g[name][k] for k in NUMBER_GROUPS]
        shares = [100 * c / g[name]["total"] for c in counts]
        bars = ax.bar(x + (i - 0.5) * w, shares, w * 0.94, color=color,
                      label=f"{name.capitalize()} rows (n = {g[name]['total']:,})")
        for b, k, c, s in zip(bars, NUMBER_GROUPS, counts, shares):
            cx = b.get_x() + b.get_width() / 2
            if k == headline[name]:
                # Beside the tall bar, in the slot of the other class's near-zero bar.
                side = w if name == "benign" else -w
                ax.text(cx + side, 55, f"{s:.2f}%\n{c:,} of\n{g[name]['total']:,}", ha="center",
                        va="center", fontsize=20, color=INK, fontweight="bold")
            else:
                ax.text(cx, s + 1.5, f"{c:,}", ha="center", va="bottom", fontsize=17, color=INK_2)
    ax.set_xticks(x, ["Number = 10", "Number = 100", "Any other value"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of the class's rows (%)")
    ax.set_xlabel("Number (packets per capture window), raw test split; small labels are row counts",
                  fontsize=17, color=INK_2, labelpad=14)
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, "leakage_number.png", out_dir)


def fig_calibration(out_dir):
    import numpy as np
    plt = _plt()
    cal, target = calibration(), target_far()
    fig, ax = _new(plt, "Calibration equalises the false-alarm rate")
    x = np.arange(2)
    w = 0.36
    groups = (("uncalibrated", "Default (h = 5, L = 3)"), ("calibrated", "Calibrated"))
    for i, (det, color) in enumerate((("CUSUM", BLUE), ("EWMA", ORANGE))):
        vals = [cal[(det, s)][1] for s, _ in groups]
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=color, label=det)
        _inside_labels(ax, bars, [f"{v:.3f}\n{cal[(det, s)][0]}" for v, (s, _) in zip(vals, groups)])
    ax.axhline(target, color=INK_2, ls="--", lw=2)
    ax.set_ylim(0, 1.3)
    ax.text(1.62, target, f"target\n{target:.1f}", va="center", ha="left", fontsize=17, color=INK_2)
    ax.set_xticks(x, [g for _, g in groups])
    ax.set_xlim(-0.6, 1.95)
    ax.set_ylabel("False alarms per 1000 windows")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    return _save(fig, "drift_calibration.png", out_dir)


def fig_scenario_c(out_dir):
    plt = _plt()
    d, seeds = scenario_c_delays(), n_seeds()
    fig, ax = _new(plt, "Scenario C (gradual benign drift): a tie")
    dets = ["CUSUM", "EWMA"]
    means = [d[k][1] for k in dets]
    stds = [d[k][2] for k in dets]
    bars = ax.bar([f"CUSUM ({d['CUSUM'][0]})", f"EWMA ({d['EWMA'][0]})"], means, width=0.5,
                  color=[BLUE, ORANGE], yerr=stds, capsize=12, error_kw={"lw": 2, "ecolor": INK_2})
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.35)
    for b, m, s, k in zip(bars, means, stds, dets):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.8, f"{m:.2f} ± {s:.2f}  (missed {d[k][3]})",
                ha="center", va="bottom", fontsize=19, color=INK)
    ax.set_ylabel("Detection delay (windows)")
    ax.set_xlabel(f"Mean ± 1 std over {seeds} seeds, both calibrated to the same false-alarm rate",
                  fontsize=17, color=INK_2, labelpad=14)
    return _save(fig, "drift_scenario_c_delay.png", out_dir)


def fig_shap_top10(out_dir):
    plt = _plt()
    rows, n_rows = shap_top10(), shap_rows()
    fig, ax = _new(plt, "Top 10 features by mean |SHAP|")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=1)
    names = [f for f, _, _ in rows][::-1]
    vals = [v for _, v, _ in rows][::-1]
    bars = ax.barh(names, vals, height=0.7, color=BLUE)
    for b, (_, v, s) in zip(bars, rows[::-1]):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}  ({s:.1%})", va="center", fontsize=17, color=INK)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel(f"Mean |SHAP| on {n_rows:,} test rows (log-odds of Mirai); share of total in brackets")
    return _save(fig, "shap_top10.png", out_dir)


def fig_repeatability(out_dir):
    import numpy as np
    plt = _plt()
    rep = repeatability()
    fig, ax = _new(plt, "Same row, 3 explanations: SHAP repeats, LIME does not")
    classes = ["benign", "Mirai"]
    x = np.arange(len(classes))
    w = 0.36
    for i, (method, color) in enumerate((("SHAP", BLUE), ("LIME", ORANGE))):
        means = [rep[(method, c)][0] for c in classes]
        stds = [rep[(method, c)][1] for c in classes]
        bars = ax.bar(x + (i - 0.5) * w, means, w * 0.94, color=color, label=method, yerr=stds, capsize=8,
                      error_kw={"lw": 2, "ecolor": INK_2})
        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, 0.04, f"{m:.3f}\n± {s:.3f}", ha="center", va="bottom",
                    fontsize=17, color="white", fontweight="bold")
    ax.set_xticks(x, ["Benign rows", "Mirai rows"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("E-XAI repeatability (top-5 kept)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
    return _save(fig, "repeatability.png", out_dir)


FIGURES = (fig_leakage, fig_leakage_number, fig_calibration, fig_scenario_c, fig_shap_top10, fig_repeatability)


def main(out_dir=FIG_DIR):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = [f(out_dir) for f in FIGURES]
    for p in paths:
        print(f"Saved {p}")
    return paths


if __name__ == "__main__":
    main()
