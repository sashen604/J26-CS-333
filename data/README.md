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
- **Dropped feature:** `Number` is excluded from all models. It is 10 for 99.95% of benign rows
  and 100 for 99.42% of Mirai rows, so it reflects how the captures were generated, not a
  property of the traffic.
- **Setup:** from the repo root, run `python p4_robustness/data_prep.py`.

#### Limitations

- **Provenance:** this is an unofficial third-party mirror, already reduced (caps of 200k benign
  and 50k per attack type, 39 of the original features), not the official UNB release.
- **Single attack family:** the positive class is Mirai floods only (greeth_flood, greip_flood,
  udpplain). Other botnet behaviours are not represented.
- **Trivial separability:** Mirai here is volumetric flood traffic, which separates easily
  from benign. Near-perfect accuracy is a property of this subset, not evidence of a strong
  detector. `Number` was removed as a measurement artefact, and the task stays easy without it.
- **Pending cross-checks:** replication on the official UNB source and a cross-check
  on N-BaIoT are both still to do.
