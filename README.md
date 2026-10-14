# J26-CS-333: Dual-Layer IoT Botnet Detection Framework

IT4010 Research Project, group **J26-CS-333**.

This repository holds the code for a dual-layer framework that detects IoT botnet
activity. The work is split into four pillars, and each pillar is owned by one team member.

## Pillars and ownership

| Folder           | Pillar                         | Owner |
|------------------|--------------------------------|-------|
| `p1_dataset/`    | P1: Dataset                    | _TBD_ |
| `p2_sdn/`        | P2: SDN layer                  | _TBD_ |
| `p3_fusion/`     | P3: Fusion                     | _TBD_ |
| `p4_robustness/` | P4: Robustness                 | _TBD_ |
| `shared/`        | Shared code used by all pillars | Whole team |
| `data/`          | Local datasets (not committed) | Everyone |

Each member organises the subfolders inside their own pillar folder however they see fit.

## Team conventions

1. **One branch per pillar.** Work on your own branch (for example `p1_dataset`,
   `p2_sdn`, `p3_fusion` or `p4_robustness`), not directly on `main`.
2. **Pull requests to `main`.** Changes reach `main` only through a pull request.
3. **No data files committed.** Datasets, CSVs and trained models stay out of Git.
   See [`data/README.md`](data/README.md) and `.gitignore`.
4. **`shared/` changes only by agreement.** Code in `shared/` affects every pillar, so
   agree on a change with the team before you make it.

## Repository layout

```
.
├── p1_dataset/
├── p2_sdn/
├── p3_fusion/
├── p4_robustness/
├── shared/
├── data/          # local only, contents ignored by Git
├── .gitignore
└── README.md
```

The `.gitkeep` files are empty placeholders. They're there because Git doesn't track empty folders.
