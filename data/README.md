# data/

This folder is for datasets on your own machine only. **Download datasets here. Don't commit them.**

Git ignores everything in this folder except this README and `.gitkeep`
(see the root `.gitignore`). Dataset files are too large for Git and may have licence
restrictions, so each team member downloads them separately.

When you start using a dataset, write down here where it comes from (name, link, version)
and any setup steps, so the rest of the team can get the same copy.

## Datasets

### `ciciot2023_hf/`: CICIoT2023 subset (HF mirror), used by P4

- **Source:** Hugging Face dataset [`lacg030175/CIC-IoT-2023`](https://huggingface.co/datasets/lacg030175/CIC-IoT-2023),
  config `random_3way`: `train.parquet` (53.3 MB), `test.parquet` (6.9 MB) and
  `validation.parquet` (6.9 MB).
- **This is an unofficial mirror.** The official release is from the Canadian Institute
  for Cybersecurity, UNB: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **A third party reduced it:** benign is capped at 200,000 rows, each attack type at
  50,000 rows, and only 39 of the original features are kept.
- **Licence:** CC BY 4.0. Cite Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark
  for Large-Scale Attacks in IoT Environment", *Sensors* 23(13), 2023.
- **Labelling results:** results from this data must be labelled
  **"CICIoT2023 subset (HF mirror)"**, not "CICIoT2023".
- **Replicating on the official source is still pending.**
- **Setup:** from the repo root, run `python p4_robustness/data_prep.py`.

#### Mirror is not official CICIoT2023

The mirror's columns share names with the official CICIoT2023 features but are **not the same
values**. A subset of the official data could not exceed the official maxima, and this one does:

| Feature | This mirror (test split) | Official CICIoT2023 (UNB statistics table) |
|---|---|---|
| `Number` | max 100 | max 15 |
| `ack_count` | max 100 | max 7.7 |
| `IAT` | all values under 0.11 | median about 83 million |
| `Variance` | up to about 7.2e7 | 0 to 1 |
| `Time_To_Live` | present | not an official field |

The mirror has **39 features**, built from a reorganised copy (`bencorn/CIC-IoT-2023`, `CSV/CSV/`),
not from Neto et al.'s original **46-feature** merged CSVs. No published definitions were found
for the mirror's versions of these columns.

Sources:
- Official feature statistics: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Uploader's READMEs: https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-canonical-neto
  (says the earlier versions were built from `CSV/CSV/`, while this one uses Neto's original merged CSVs) and
  https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-neto-full (says the source copy dropped 7 of
  the 46 original features).

#### Window-size leakage

**Rule:** drop every feature whose value depends on the measurement window length. In this
capture, window length differs by class: 10 packets for benign, 100 for Mirai.

**Dropped (7):** `Number`, `Tot sum`, `ack_count`, `rst_count`, `syn_count`, `fin_count`, `Tot size`
(see `EXCLUDED_FEATURES` in `p4_robustness/data_prep.py`).

**Evidence (test split):**
- `Number`, the packets per window, is 10 for 99.97% of benign rows (19,994 of 19,999) and 100 for 99.52% of Mirai
  rows (15,075 of 15,147).
  A one-split rule on it alone reached 0.9998 test accuracy.
- `ack_count`, `rst_count`, `syn_count` and `fin_count` are ≤ `Number` on every test row. They are
  counts over the window (benign max 10 / 6 / 8 / 3, Mirai max 100 / 58 / 38 / 11).
- `Tot sum / (Number × AVG)` = 1.0000 at the 1st, 50th and 99th percentiles, for both classes.
- `Tot size` is identical to `AVG` on every row. It is a duplicate, dropped for tidiness.

**Resolution leak (fixed by quantising, not dropping):** the remaining averages and shares are
means over the window's packets. Benign values are means over 10 packets, so they fall on a
0.1 grid. Mirai values are means over 100 packets, so they fall on a 0.01 grid. On the test split,
before rounding, 99.98% of benign rows (19,995 of 19,999) had all 28 such features on the 0.1 grid,
not 100%. The 4 off-grid benign rows have `Number` = 4, 9, 8 and 6, i.e. windows of fewer than
10 packets, so off-grid values follow window size, which supports the leakage explanation. The rule
"any value off the 0.1 grid → Mirai" scored **0.8134** test accuracy using precision alone.
**Fix:** the 28 features in `QUANTISED_FEATURES` (`p4_robustness/data_prep.py`) are rounded
to 1 decimal place for both classes on load. After rounding, the same rule scores **0.5690**,
exactly the majority-class baseline. `Rate`, `IAT`, `Std` and `Variance` are not on a grid and
are left unrounded.

#### Limitations

- **Not official features:** every result in `p4_robustness/results/` applies to the mirror's
  version of the features, not to CICIoT2023 as published (see "Mirror is not official CICIoT2023").
  Replication on official data is still pending, using either the UNB download or the uploader's
  `lacg030175/CIC-IoT-2023-canonical-neto` dataset.

- **Provenance:** this is an unofficial third-party mirror, already reduced (caps of 200k benign
  and 50k per attack type, 39 of the original features), not the official UNB release.
- **Single attack family:** the positive class is Mirai floods only (greeth_flood, greip_flood,
  udpplain). Other botnet behaviours are not represented.
- **Window-size leakage:** proxy accuracy before the window-size fix reflected the capture
  window size (10 vs 100 packets), not traffic behaviour. Removing `Number` alone was not enough,
  because `Tot sum` (= `Number × AVG`) and the count features still carried it.
- **Residual window dependence:** `Std` and `Variance` still depend on sample size, because
  spread estimated over 10 packets behaves differently from spread over 100. Quantising removes
  the precision clue but not other effects of the window length, such as how much 10-packet
  means vary. The proper fix is to recompute features from the PCAPs with one window length for
  both classes. That is still pending.
- **Separability:** Mirai here is volumetric flood traffic. High accuracy on this subset is not
  evidence of a strong detector.
- **Pending cross-checks:** replication on the official UNB source and a cross-check
  on N-BaIoT are both still to do.
