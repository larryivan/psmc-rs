# Experiment (Minimal)

This folder is trimmed to the minimum set for **one-command check + plotting**.

## One command

```bash
cd /Users/larryivanhan/Documents/lab/rspsmc/psmc-rs
./experiment/run.py
```

What it does:

1. Validates required experiment files and key tables.
2. Executes `experiment/notebooks/all_plots_precomputed.ipynb` headlessly.
3. Regenerates main and supplementary figures (`png/svg/pdf`).

## Kept structure

```text
experiment/
├── README.md
├── run.py
├── notebooks/
│   └── all_plots_precomputed.ipynb
├── real_data/
│   ├── HLemySub1_diploid.psmcfa
│   ├── HLhydTec1_diploid.psmcfa
│   ├── HLpelCas1_diploid.psmcfa
│   └── Papuan_highlands_diploid.psmcfa
├── assets/
│   └── ui/
│       ├── README.md
│       ├── tui_run.png         # optional for Supplementary Figure S3
│       └── html_report.png     # optional for Supplementary Figure S3
└── runs/
    ├── main_text/
    │   ├── outputs/
    │   ├── tables/
    │   └── figures/
    ├── real_data/
    │   ├── outputs/
    │   ├── tables/
    │   └── figures/
    └── supplementary/
        ├── tables/
        └── figures/
```

## Useful options

```bash
# check only
./experiment/run.py --skip-plot

# plot only
./experiment/run.py --skip-check
```
