# Time-varying global integration of the FTSE/JSE All Share — code and data

Reproduces every table and figure in the thesis. Frozen data hashes are in `DATA_FREEZE.txt`; the estimator harness results are in `VALIDATION.md`.

## Layout
```
data/thesis_data.csv            main weekly panel, Jan 1999 – Sep 2026, 44 series (hash in DATA_FREEZE.txt)
data/thesis_data_domestic.csv   domestic-risk panel from Dec 2002 (adds ALBIAY, 5Y CDS, MSCI EM+FM Africa ex SA)
01_validate.py                  estimator harness — run first
02_run_specs.py                 model + specification runner (screen, full runs, tau sweep, drop-one)
03_figures.py                   thesis figures from results/
results/                        outputs of 02 (committed from the frozen run)
figures/                        outputs of 03
```

## Reproduce (in order)
```
pip install -r requirements.txt
python 01_validate.py                                             # must print ALL PASS
python 02_run_specs.py --screen --full --alpha 5.6 --tau 52 104 156
python 02_run_specs.py --drop-one baseline_mm --alpha 5.6 --tau 104 --targets JALSH RESI20 FINI15
python 02_run_specs.py --csv data/thesis_data_domestic.csv --out results_domestic --full --alpha 5.6 --tau 104 --targets JALSH FINI15 JGENR
python 03_figures.py
```
Roughly 2.5 hours total on a laptop. Runs are resumable — re-invoking skips anything already in `results/`.

## Model
Weighted ridge regression of weekly log returns on named global channels, with observation weights following the flexible-probabilities framework (Meucci 2010; exponential time decay, half-life τ = 104 weeks) and ridge penalty α = 5.6 (median of the leave-one-out-selected path). Every estimation date uses the expanding history to that date; regressors are contemporaneous. Presented specification `baseline_mm`: MSCI World Metals & Mining, MSCI EM, MSCI World Consumer Staples, USDZAR.

## Figure / table map
| Thesis | Source |
|---|---|
| Fig 1 loadings, R², ENS | `03_figures.py: fig_loadings_r2_ens` |
| Fig 2 signed contributions | `fig_contrib_signed` |
| Fig 3 share of influence | `fig_share_of_influence` |
| Fig 4 sub-index validation | `fig_subindex_validation` |
| Fig 5 decomposition gain | `fig_decomposition_gain` |
| Fig 6 nested ladder | `fig_ladder` |
| Fig 7 half-life sensitivity | `fig_tau_sensitivity` |
| Table: specification comparison | `results/tau_104/summary.csv` |
| Table: half-life sweep | `results/tau_sensitivity_LOO.csv` |
| Table: domestic replication | `results_domestic/summary.csv` |
| Table: drop-one ΔR² | `results/tau_104/<target>/baseline_mm/drop_one_summary.csv` |
