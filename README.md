# J26-CS-333: Dual-Layer IoT Botnet Detection Framework

IT4010 Research Project, group **J26-CS-333**.

This repository holds the code for a dual-layer framework that detects IoT botnet
activity. The work is split into four pillars, and each pillar is owned by one team member.

## Pillars and ownership

| Folder           | Pillar                          | Owner      | Branch                  |
|------------------|---------------------------------|------------|-------------------------|
| `p1_dataset/`    | P1: Dataset                     | Sunera     | `feature/p1-dataset`    |
| `p2_sdn/`        | P2: SDN layer                   | Sasmitha   | `feature/p2-sdn`        |
| `p3_fusion/`     | P3: Fusion                      | Janith     | `feature/p3-fusion`     |
| `p4_robustness/` | P4: Robustness                  | Kavish     | `feature/p4-robustness` |
| `shared/`        | Shared code used by all pillars | Whole team | (none; changes by agreement) |
| `data/`          | Local datasets (not committed)  | Everyone   | (none)                  |

Each member organises the subfolders inside their own pillar folder however they see fit.

## Team conventions

1. **One branch per pillar.** Work on your own branch, not directly on `main`. Name it
   `feature/pN-name`, where `N` is the pillar number, for example `feature/p4-robustness`.
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
