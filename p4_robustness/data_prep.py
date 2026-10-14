"""Download and prepare the CICIoT2023 subset used by the Pillar 4 proxy model.

PROVENANCE (read before using any result from this data)
--------------------------------------------------------
- Source: Hugging Face dataset ``lacg030175/CIC-IoT-2023``, config ``random_3way``
  (train / test / validation Parquet files, about 67 MB in total).
  https://huggingface.co/datasets/lacg030175/CIC-IoT-2023
- This is an UNOFFICIAL MIRROR. The official CICIoT2023 release is from the
  Canadian Institute for Cybersecurity, UNB:
  https://www.unb.ca/cic/datasets/iotdataset-2023.html
- A third party reduced the data: benign is capped at 200,000 rows, each attack
  type at 50,000 rows, and only 39 of the original features are kept.
- Licence: CC BY 4.0. Cite Neto et al., "CICIoT2023: A Real-Time Dataset and
  Benchmark for Large-Scale Attacks in IoT Environment", Sensors 23(13), 2023.
- Results from this data must be labelled "CICIoT2023 subset (HF mirror)",
  NOT "CICIoT2023".
- Replicating on the official source is still pending.

LABEL DEFINITION
----------------
Only benign rows and Mirai rows are kept. Positive class (1) = Mirai botnet,
negative class (0) = benign. DDoS, DoS, Recon, Spoofing, Web-based and
BruteForce rows are dropped. No rebalancing or resampling is done here.

Files are stored in data/ciciot2023_hf/, which is gitignored.
"""
from pathlib import Path
import urllib.request

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "ciciot2023_hf"

HF_BASE = "https://huggingface.co/datasets/lacg030175/CIC-IoT-2023/resolve/main/random_3way"
SPLITS = ("train", "test", "validation")

# Columns that describe the label. These must never be used as model features.
LABEL_COLUMNS = ("label", "Label", "attack_class")
TARGET = "is_mirai"

# Features dropped permanently. 'Number' is 10 for 99.95% of benign rows and 100 for
# 99.42% of Mirai rows in this subset, so it reflects how the captures were generated,
# not a property of the traffic. A single split on it alone reached 0.9998 test accuracy.
EXCLUDED_FEATURES = ("Number",)


def split_url(split):
    return f"{HF_BASE}/{split}-00000-of-00001.parquet"


def split_path(split, data_dir=DATA_DIR):
    return Path(data_dir) / f"{split}.parquet"


def download(data_dir=DATA_DIR, force=False):
    """Download the three Parquet splits into data_dir. Skips files already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dest = split_path(split, data_dir)
        if dest.exists() and not force:
            print(f"{split}: already downloaded ({dest.stat().st_size:,} bytes)")
            continue
        print(f"{split}: downloading {split_url(split)}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(split_url(split), tmp)
        tmp.replace(dest)
        print(f"{split}: saved {dest.stat().st_size:,} bytes to {dest}")


def filter_mirai_vs_benign(df):
    """Keep only benign and Mirai rows and add the binary TARGET column (1 = Mirai)."""
    group = df["attack_class"]
    keep = group.isin(["Benign", "Mirai"])
    out = df.loc[keep].copy()
    out[TARGET] = (out["attack_class"] == "Mirai").astype(int)
    return out.reset_index(drop=True)


def feature_columns(df):
    """Feature columns: everything except label columns, TARGET and EXCLUDED_FEATURES."""
    excluded = set(LABEL_COLUMNS) | {TARGET} | set(EXCLUDED_FEATURES)
    return [c for c in df.columns if c not in excluded]


def load_split(split, data_dir=DATA_DIR):
    """Load one split, already filtered to Mirai vs benign."""
    path = split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python p4_robustness/data_prep.py` first.")
    return filter_mirai_vs_benign(pd.read_parquet(path))


def class_counts(df):
    """Return (benign_count, mirai_count) for a filtered split."""
    mirai = int(df[TARGET].sum())
    return len(df) - mirai, mirai


if __name__ == "__main__":
    download()
    for split in SPLITS:
        df = load_split(split)
        benign, mirai = class_counts(df)
        total = benign + mirai
        print(f"{split:10} benign={benign:,}  mirai={mirai:,}  total={total:,}  "
              f"mirai share={mirai / total:.3f}")
