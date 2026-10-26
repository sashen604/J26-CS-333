"""Window-size leakage audit for Pillar 4: reproduces the evidence in data/README.md.

Run from the repo root:  python p4_robustness/experiments/leakage_audit.py

Works on the RAW mirror data (filtered to Mirai vs benign, NOT quantised), test split:
1. `Number` (packets per window): benign vs Mirai value distribution (10 vs 100).
2. Tot sum / (Number x AVG) at the 1st, 50th and 99th percentiles, per class.
3. Share of rows where each count feature is <= Number, with per-class maxima.
4. Accuracy of the rule "any QUANTISED_FEATURES value off the 0.1 grid -> Mirai",
   before and after quantisation, against the majority-class baseline, and the
   Number of the few benign rows that are off the grid.

Each figure is compared with the value documented in data/README.md (README_EXPECTED).
The script exits with status 1 if any figure differs.

Data: "CICIoT2023 subset (HF mirror)"; see data/README.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4_DIR))

import data_prep  # noqa: E402

COUNT_FEATURES = ("ack_count", "rst_count", "syn_count", "fin_count")
PERCENTILES = (1, 50, 99)
BENIGN_WINDOW, MIRAI_WINDOW = 10, 100
GRID_TOL = 1e-6  # |10x - round(10x)| above this counts as off the 0.1 grid

# Figures documented in data/README.md (test split), at the precision the README states them.
README_EXPECTED = {
    "number_benign_share_10": "99.97%",
    "number_mirai_share_100": "99.52%",
    "number_stump_accuracy": "0.9998",
    "totsum_ratio_benign": ["1.0000", "1.0000", "1.0000"],
    "totsum_ratio_mirai": ["1.0000", "1.0000", "1.0000"],
    "count_le_number_share": {f: "100.00%" for f in COUNT_FEATURES},
    "count_max_benign": {"ack_count": 10, "rst_count": 6, "syn_count": 8, "fin_count": 3},
    "count_max_mirai": {"ack_count": 100, "rst_count": 58, "syn_count": 38, "fin_count": 11},
    "tot_size_equals_avg_share": "100.00%",
    "benign_rows_on_grid": "19,995 of 19,999",
    "off_grid_benign_numbers": [4, 6, 8, 9],
    "grid_rule_accuracy_before": "0.8134",
    "grid_rule_accuracy_after": "0.5690",
    "majority_baseline": "0.5690",
}


def load_raw(split="test", data_dir=data_prep.DATA_DIR):
    """One split filtered to Mirai vs benign, WITHOUT quantisation (the raw mirror values)."""
    path = data_prep.split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python p4_robustness/data_prep.py` first.")
    return data_prep.filter_mirai_vs_benign(pd.read_parquet(path))


def _classes(df):
    y = df[data_prep.TARGET]
    return df[y == 0], df[y == 1]


def number_distribution(df, top=3):
    """Share of benign rows with Number == 10 and of Mirai rows with Number == 100, plus top values."""
    benign, mirai = _classes(df)
    return {
        "benign_share_10": float((benign["Number"] == BENIGN_WINDOW).mean()),
        "mirai_share_100": float((mirai["Number"] == MIRAI_WINDOW).mean()),
        "benign_top": benign["Number"].value_counts(normalize=True).head(top).to_dict(),
        "mirai_top": mirai["Number"].value_counts(normalize=True).head(top).to_dict(),
        "n_benign": len(benign),
        "n_mirai": len(mirai),
    }


def number_stump_accuracy(train, test):
    """Test accuracy of a single split on Number, fitted on train. Returns (accuracy, threshold)."""
    from sklearn.tree import DecisionTreeClassifier
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)
    stump.fit(train[["Number"]], train[data_prep.TARGET])
    return float(stump.score(test[["Number"]], test[data_prep.TARGET])), float(stump.tree_.threshold[0])


def totsum_ratio_percentiles(df, percentiles=PERCENTILES):
    """Tot sum / (Number x AVG) at the given percentiles, per class."""
    ratio = df["Tot sum"] / (df["Number"] * df["AVG"])
    y = df[data_prep.TARGET]
    return {name: np.percentile(ratio[y == c], percentiles).tolist()
            for c, name in ((0, "benign"), (1, "mirai"))}


def count_feature_bounds(df, features=COUNT_FEATURES):
    """Per count feature: share of rows with value <= Number, and the benign and Mirai maxima."""
    benign, mirai = _classes(df)
    return {f: {"le_number_share": float((df[f] <= df["Number"]).mean()),
                "max_benign": float(benign[f].max()),
                "max_mirai": float(mirai[f].max())}
            for f in features}


def tot_size_equals_avg_share(df):
    return float((df["Tot size"] == df["AVG"]).mean())


def off_grid_rows(df, features=data_prep.QUANTISED_FEATURES, tol=GRID_TOL):
    """Boolean per row: True if any of `features` has a value off the 0.1 grid."""
    v = df[list(features)].to_numpy(dtype=float) * 10
    return np.abs(v - np.rint(v)).max(axis=1) > tol


def grid_rule_accuracy(df):
    """Accuracy of the rule "any value off the 0.1 grid -> Mirai, else benign"."""
    pred = off_grid_rows(df).astype(int)
    return float((pred == df[data_prep.TARGET].to_numpy()).mean())


def majority_baseline(df):
    share = float(df[data_prep.TARGET].mean())
    return max(share, 1 - share)


def run_audit(data_dir=data_prep.DATA_DIR):
    """All audit figures from the raw test split (and train, for the Number stump)."""
    test = load_raw("test", data_dir)
    train = load_raw("train", data_dir)
    benign, _ = _classes(test)
    stump_acc, stump_thr = number_stump_accuracy(train, test)
    return {
        "number": number_distribution(test),
        "stump": (stump_acc, stump_thr),
        "totsum_ratio": totsum_ratio_percentiles(test),
        "counts": count_feature_bounds(test),
        "tot_size_equals_avg": tot_size_equals_avg_share(test),
        "benign_rows_on_grid": (int((~off_grid_rows(benign)).sum()), len(benign)),
        "off_grid_benign_numbers": sorted(int(n) for n in benign.loc[off_grid_rows(benign), "Number"]),
        "grid_before": grid_rule_accuracy(test),
        "grid_after": grid_rule_accuracy(data_prep.quantise(test)),
        "majority": majority_baseline(test),
    }


def as_readme_figures(r):
    """Format audit results at the README's precision, keyed like README_EXPECTED."""
    on, n = r["benign_rows_on_grid"]
    return {
        "number_benign_share_10": f"{r['number']['benign_share_10']:.2%}",
        "number_mirai_share_100": f"{r['number']['mirai_share_100']:.2%}",
        "number_stump_accuracy": f"{r['stump'][0]:.4f}",
        "totsum_ratio_benign": [f"{v:.4f}" for v in r["totsum_ratio"]["benign"]],
        "totsum_ratio_mirai": [f"{v:.4f}" for v in r["totsum_ratio"]["mirai"]],
        "count_le_number_share": {f: f"{c['le_number_share']:.2%}" for f, c in r["counts"].items()},
        "count_max_benign": {f: int(c["max_benign"]) for f, c in r["counts"].items()},
        "count_max_mirai": {f: int(c["max_mirai"]) for f, c in r["counts"].items()},
        "tot_size_equals_avg_share": f"{r['tot_size_equals_avg']:.2%}",
        "benign_rows_on_grid": f"{on:,} of {n:,}",
        "off_grid_benign_numbers": r["off_grid_benign_numbers"],
        "grid_rule_accuracy_before": f"{r['grid_before']:.4f}",
        "grid_rule_accuracy_after": f"{r['grid_after']:.4f}",
        "majority_baseline": f"{r['majority']:.4f}",
    }


def compare_with_readme(r, expected=README_EXPECTED):
    """List of (key, measured, documented) for every figure that differs from data/README.md."""
    measured = as_readme_figures(r)
    return [(k, measured[k], v) for k, v in expected.items() if measured[k] != v]


def print_report(r):
    num, (acc, thr) = r["number"], r["stump"]
    print("[1] Number (packets per window), test split")
    print(f"    benign: {num['benign_share_10']:.2%} of {num['n_benign']:,} rows have Number = 10")
    print(f"    Mirai:  {num['mirai_share_100']:.2%} of {num['n_mirai']:,} rows have Number = 100")
    print(f"    one split on Number (threshold {thr:g}, fitted on train): test accuracy {acc:.4f}")
    print("[2] Tot sum / (Number x AVG), percentiles " + " / ".join(f"p{p}" for p in PERCENTILES))
    for name, label in (("benign", "benign"), ("mirai", "Mirai ")):
        print(f"    {label}: " + " / ".join(f"{v:.4f}" for v in r["totsum_ratio"][name]))
    print(f"    Tot size == AVG on {r['tot_size_equals_avg']:.2%} of rows (a duplicate)")
    print("[3] Count features vs Number")
    for f, c in r["counts"].items():
        print(f"    {f:9}  <= Number on {c['le_number_share']:.2%} of rows; "
              f"max benign {c['max_benign']:g}, max Mirai {c['max_mirai']:g}")
    on, n = r["benign_rows_on_grid"]
    print("[4] Rule: any of the 28 QUANTISED_FEATURES off the 0.1 grid -> Mirai")
    print(f"    benign rows with all 28 on the grid: {on:,} of {n:,} ({on / n:.2%})")
    print(f"    Number of the {n - on} off-grid benign rows: "
          + ", ".join(str(v) for v in r["off_grid_benign_numbers"]) + " (all below 10)")
    print(f"    accuracy BEFORE quantising: {r['grid_before']:.4f}")
    print(f"    accuracy AFTER quantising:  {r['grid_after']:.4f}  (majority-class baseline {r['majority']:.4f})")


def main():
    r = run_audit()
    print_report(r)
    diffs = compare_with_readme(r)
    if diffs:
        print("\nMISMATCH with data/README.md:")
        for k, got, want in diffs:
            print(f"    {k}: measured {got}, README {want}")
        return 1
    print("\nAll figures match data/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
