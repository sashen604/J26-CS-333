"""Explainability experiment for Pillar 4: global SHAP summary and SHAP vs LIME stability.

Run from the repo root:  python p4_robustness/experiments/xai_experiment.py

1. Global SHAP summary on 1,000 random test rows -> results/shap_summary.png, and the
   top 10 features by mean |SHAP|.
2. Perturbation stability (xai/metrics.py; per CCS 2022, Gan et al.) for SHAP and LIME on
   100 test rows (50 benign, 50 Mirai), 10 valid perturbations each, k = 5, fixed seeds.
   Reported on all perturbations and again excluding those that flipped the predicted class.
3. E-XAI repeatability (Arreche et al.) on the same 100 unchanged rows, N = 3 runs.
4. results/xai_results.md with the tables and a Findings section.

Data: "CICIoT2023 subset (HF mirror)"; see data/README.md. Model: XGBoost proxy, not
P2's AI model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4_DIR))

import data_prep  # noqa: E402
from proxy_model import ProxyFusionModel  # noqa: E402
from xai.explainers import (lime_explain, make_lime_explainer, make_shap_explainer,  # noqa: E402
                            shap_explain)
from xai.metrics import (DEFAULT_REL_EPS, exai_repeatability, explanation_stability,  # noqa: E402
                         perturb_row)

RESULTS_DIR = P4_DIR / "results"
N_SUMMARY = 1000
N_PER_CLASS = 50
N_PERTURB = 10
N_REPEAT = 3  # E-XAI uses N = 3 repeated runs
K = 5
N_LIME_REFERENCE = 5000
LIME_SAMPLES = 5000
SEED = 0


def sample_rows(df, n, seed):
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def mean_abs_shap(values, feature_names):
    s = pd.Series(np.abs(values).mean(axis=0), index=feature_names)
    return s.sort_values(ascending=False)


def summarise(rows, prefix=""):
    """(mean, sample std, n non-NaN rows) per metric; NaN rows (e.g. no kept perturbations) skipped."""
    df = pd.DataFrame(rows)
    return {m: (float(df[prefix + m].mean()), float(df[prefix + m].std(ddof=1)), int(df[prefix + m].notna().sum()))
            for m in ("jaccard", "spearman", "intersection")}


def stability_table(stability):
    lines = ["| Method | Class | Jaccard | Spearman | Intersection |", "|---|---|---|---|---|"]
    for method in ("SHAP", "LIME"):
        for cls_name in ("benign", "Mirai"):
            s = stability[(method, cls_name)]
            cell = []
            for m in ("jaccard", "spearman", "intersection"):
                mean, std, n = s[m]
                cell.append(f"{mean:.3f} ± {std:.3f}" + (f" (n={n})" if n < N_PER_CLASS else ""))
            lines.append(f"| {method} | {cls_name} | " + " | ".join(cell) + " |")
    return lines


def save_summary_plot(shap_values, X, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    shap.summary_plot(shap_values, X, max_display=15, show=False)
    plt.title("SHAP summary, 1,000 test rows (positive = toward Mirai)\n"
              "CICIoT2023 subset (HF mirror), proxy model", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = ProxyFusionModel.load()
    test = data_prep.load_split("test")
    train = data_prep.load_split("train")
    features = model.feature_names
    shap_explainer = make_shap_explainer(model)

    # 1. Global SHAP summary
    summary_rows = sample_rows(test, N_SUMMARY, SEED)
    sv = shap_explain(model, summary_rows, shap_explainer)
    ranking = mean_abs_shap(sv.values, features)
    share = ranking / ranking.sum()
    save_summary_plot(sv.values, summary_rows[features], RESULTS_DIR / "shap_summary.png")
    n_mirai_summary = int(summary_rows[data_prep.TARGET].sum())
    print("Top 10 by mean |SHAP|:")
    print(pd.DataFrame({"mean_abs_shap": ranking, "share": share}).head(10).round(4).to_string())

    # 2. Stability
    lime_ref = sample_rows(train, N_LIME_REFERENCE, SEED)
    lime_explainer = make_lime_explainer(model, lime_ref, SEED)
    benign = sample_rows(test[test[data_prep.TARGET] == 0], N_PER_CLASS, SEED)
    mirai = sample_rows(test[test[data_prep.TARGET] == 1], N_PER_CLASS, SEED)

    def shap_fn(r):
        return shap_explain(model, r, shap_explainer).values[0]

    def lime_fn(r):
        return lime_explain(model, r, lime_ref, k=K, explainer=lime_explainer,
                            num_samples=LIME_SAMPLES, seed=SEED).weights

    # E-XAI repeatability: LIME runs WITHOUT per-call reseeding, so each run draws fresh samples.
    # This separate explainer is seeded once at construction only so the whole script is reproducible.
    lime_free = make_lime_explainer(model, lime_ref, SEED)

    def lime_free_fn(r):
        return lime_explain(model, r, lime_ref, k=K, explainer=lime_free,
                            num_samples=LIME_SAMPLES, seed=None).weights

    def keep_same_class(row):
        p0 = model.predict(row)[0]
        return lambda xp: model.predict(xp)[0] == p0

    stability, stability_kept, repeat, flips, n_perturbed = {}, {}, {}, {}, {}
    for cls_name, rows in (("benign", benign), ("Mirai", mirai)):
        for method, fn, rep_fn in (("SHAP", shap_fn, shap_fn), ("LIME", lime_fn, lime_free_fn)):
            res = [explanation_stability(fn, rows.iloc[[i]], N_PERTURB, K, seed=SEED + i,
                                         keep=keep_same_class(rows.iloc[[i]]))
                   for i in range(len(rows))]
            stability[(method, cls_name)] = summarise(res)
            stability_kept[(method, cls_name)] = summarise(res, prefix="kept_")
            # Predicted-class flips depend only on the perturbations, which are identical for both methods.
            flips[cls_name] = sum(N_PERTURB - r["n_kept"] for r in res)
            n_perturbed[cls_name] = len(rows) * N_PERTURB
            rep = [exai_repeatability(rep_fn, rows.iloc[[i]], N_REPEAT, K) for i in range(len(rows))]
            repeat[(method, cls_name)] = (float(np.mean(rep)), float(np.std(rep, ddof=1)))
            s, sk = stability[(method, cls_name)], stability_kept[(method, cls_name)]
            print(f"{method} {cls_name}: jaccard {s['jaccard'][0]:.3f} ± {s['jaccard'][1]:.3f}, "
                  f"spearman {s['spearman'][0]:.3f} ± {s['spearman'][1]:.3f} (n={s['spearman'][2]}), "
                  f"intersection {s['intersection'][0]:.3f} ± {s['intersection'][1]:.3f} | kept-only: "
                  f"jaccard {sk['jaccard'][0]:.3f} ± {sk['jaccard'][1]:.3f} (n={sk['jaccard'][2]}), "
                  f"spearman {sk['spearman'][0]:.3f} ± {sk['spearman'][1]:.3f}, "
                  f"intersection {sk['intersection'][0]:.3f} ± {sk['intersection'][1]:.3f} | "
                  f"E-XAI repeatability {repeat[(method, cls_name)][0]:.3f} ± {repeat[(method, cls_name)][1]:.3f}")
    print("prediction flips under perturbation:", {c: f"{flips[c]}/{n_perturbed[c]}" for c in flips})
    gain = model.model.get_booster().get_score(importance_type="gain")
    gain_share = {f: v / sum(gain.values()) for f, v in gain.items()}
    rows_with_flip = {}
    for cls_name, rows in (("benign", benign), ("Mirai", mirai)):
        n = 0
        for i in range(len(rows)):
            row, rng = rows.iloc[[i]], np.random.default_rng(SEED + i)
            p0 = model.predict(row)[0]
            n += any(model.predict(perturb_row(row, rng))[0] != p0 for _ in range(N_PERTURB))
        rows_with_flip[cls_name] = n

    # 3. Report
    top = ranking.index[0]
    lines = [
        "# Explainability results: SHAP vs LIME (Pillar 4)",
        "",
        "Features: HF mirror reconstruction, not official CICIoT2023 — see data/README.md.",
        "",
        "Generated by `python p4_robustness/experiments/xai_experiment.py`. Do not edit by hand.",
        "",
        "**Data:** CICIoT2023 subset (HF mirror), Mirai vs benign, with window-size leakage features excluded "
        "and 28 features quantised (see `data/README.md`). **Model:** XGBoost proxy, which is not P2's AI model. "
        "The proxy's near-perfect accuracy is not a detection result.",
        "",
        "## Global SHAP: top 10 features by mean |SHAP|",
        "",
        f"{N_SUMMARY:,} random test rows (seed {SEED}; {N_SUMMARY - n_mirai_summary} benign, {n_mirai_summary} Mirai). "
        "SHAP values are in log-odds of Mirai: positive pushes toward Mirai, negative toward benign. "
        "Share = this feature's mean |SHAP| / the sum over all features.",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| | Share |",
        "|---|---|---|---|",
    ]
    lines += [f"| {i + 1} | {f} | {ranking[f]:.4f} | {share[f]:.3f} |" for i, f in enumerate(ranking.index[:10])]
    lines += [
        "",
        f"**Dominance:** `{top}` accounts for {share[top]:.1%} of total mean |SHAP|. The next feature, "
        f"`{ranking.index[1]}`, accounts for {share[ranking.index[1]]:.1%}.",
        "",
        "Plot: `shap_summary.png`.",
        "",
        "## Explanation stability under small valid perturbations",
        "",
        f"{N_PER_CLASS} benign + {N_PER_CLASS} Mirai test rows (seed {SEED}), {N_PERTURB} perturbations per row, "
        f"top-k with k = {K}. Each perturbation multiplies every continuous feature by an independent factor in "
        f"[{1 - DEFAULT_REL_EPS:g}, {1 + DEFAULT_REL_EPS:g}]. Protocol codes and 0–1 shares are not perturbed, known "
        "upper bounds are clipped, and quantised features are re-rounded to 0.1. LIME uses a fixed seed and "
        f"{LIME_SAMPLES:,} samples per explanation, with sampling statistics from {N_LIME_REFERENCE:,} train rows. "
        "Definitions are in `xai/metrics.py`.",
        "",
        "**Citation:** perturbation stability per Gan et al., \"Is your explanation stable?\": A Robustness "
        "Evaluation Framework for Feature Attribution (ACM CCS 2022). The Intersection score uses E-XAI's formula "
        "(Arreche et al., IEEE Access 2024, Sec. C.4), adapted to perturbed inputs. Jaccard and Spearman are our "
        "additions. (The CCS 2022 paper's own stability metric, mstd, is not computed here.)",
        "",
        "- **Jaccard:** mean overlap of the top-5 set with the unperturbed row's top-5.",
        "- **Spearman:** mean rank correlation of all attributions with the unperturbed row's attributions.",
        "- **Intersection:** the share of the unperturbed top-5 that stays in the top-5 under every perturbation.",
        "",
        "Values are mean ± std over rows (sample std). All metrics are 1.0 for a perfectly stable explanation.",
        "",
        "### All perturbations",
        "",
    ]
    lines += stability_table(stability)
    lines += [
        "",
        "### Excluding perturbations that flipped the predicted class",
        "",
        "These are the same perturbations, minus any that changed the model's predicted class for that row. "
        "Rows where every perturbation flipped would be dropped, and n is shown when fewer than 50 rows remain.",
        "",
    ]
    lines += stability_table(stability_kept)
    lines += [
        "",
        "The model's predicted class changed in "
        + ", ".join(f"{flips[c]} of {n_perturbed[c]} {c}" for c in flips)
        + " perturbed rows ("
        + ", ".join(f"{rows_with_flip[c]} of {N_PER_CLASS} {c} rows had at least one flip" for c in flips) + ").",
        "",
        "## E-XAI repeatability (Arreche et al.)",
        "",
        f"E-XAI's original stability (IEEE Access 2024, Sec. C.4): explain the same {2 * N_PER_CLASS} UNCHANGED rows "
        f"N = {N_REPEAT} times each and report |X_1 ∩ X_2 ∩ X_3| / k for the top-{K}. LIME runs with its normal "
        "random sampling: it is not reseeded between runs, and the explainer is seeded once at construction so the "
        "script is reproducible. SHAP TreeExplainer is deterministic, so its score is 1.0 by construction.",
        "",
        "| Method | Class | Repeatability |",
        "|---|---|---|",
    ]
    lines += [f"| {m} | {c} | {repeat[(m, c)][0]:.3f} ± {repeat[(m, c)][1]:.3f} |"
              for m in ("SHAP", "LIME") for c in ("benign", "Mirai")]
    std_rank = list(ranking.index).index("Std") + 1
    lines += [
        "",
        "## Findings",
        "",
        f"- **`{top}` leads SHAP at {share[top]:.1%} vs {gain_share.get(top, 0):.1%} of XGBoost gain.** Gain adds up "
        "the loss reduction from every split on a feature, so the feature chosen for the large early splits collects "
        "most of it. SHAP divides each individual prediction among all the features that moved it, so features used "
        f"in later, smaller splits (`{ranking.index[1]}`, `{ranking.index[2]}`) still receive substantial credit.",
        f"- **Flip asymmetry:** under ±{DEFAULT_REL_EPS:.0%} valid perturbations, {flips['benign']}/{n_perturbed['benign']} "
        f"benign vs {flips['Mirai']}/{n_perturbed['Mirai']} Mirai perturbations changed the predicted class. Some benign "
        "predictions sit close to the decision boundary, while Mirai predictions do not move.",
        f"- **`Std` is the #{std_rank} most influential feature ({share['Std']:.1%} of mean |SHAP|)** and keeps its "
        "sample-size dependence (spread over 10 vs 100 packets). This is a documented limitation in "
        "`data/README.md`, pending the PCAP feature recompute.",
        "",
    ]
    (RESULTS_DIR / "xai_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote xai_results.md and shap_summary.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
