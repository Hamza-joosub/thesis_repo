# Decomposing the Rand Hedge: code and results

Code, model output and figures for *Decomposing the Rand Hedge: A Time-Varying Factor Model of the FTSE/JSE Top 40 Estimated with Flexible Probabilities*, M. H. Joosub, ECO4106Z long paper, School of Economics, University of Cape Town, November 2026. Supervisor: Dr Godfrey Ndlovu.

## Model

Weighted ridge regression of the Top 40's weekly log return on three global channels: MSCI World Metals and Mining, MSCI Emerging Markets and USDZAR (specification `em_ladder_2`). Observations are weighted by flexible probabilities with exponential time decay, half-life τ = 104 weeks, and the ridge penalty is α = 5.6. Each estimate uses the history up to its date; the first is made in October 2003. The presented model's weekly loadings, contributions and fit are in `results/tau_104/TOP40/em_ladder_2/`, and the half-life checks in `results/tau_52/` and `results/tau_156/`.

## Data

The price and index-weight data are from Bloomberg and are licensed, so they are not in this repository. `DATA.md` lists every ticker and field, and the file names the scripts expect.

## Scripts, in run order

| Script | What it does |
|---|---|
| `00_transform.py` | Builds the weekly panel (levels, log and simple differences) from a raw Bloomberg export |
| `01_validate.py` | Estimator checks, including the Matthews (2024) probabilities and effective number of scenarios; run first |
| `02_run_specs.py` | Estimates the specifications across indices, half-lives and penalties; writes `results/` |
| `03_figures.py` | Figures 3 to 5 |
| `04_robustness.py` | The Table 5 checks (`results_robustness/`, one CSV per check) and Figures 1 and 6 |
| `05_stock_translation.py` | Currency translation of each dual-listed pair (`stock_translation.csv`, `stock_translation_baskets.csv`) |
| `06_foreign_price_check.py` | Foreign-price check of the dual-listed constituents (`foreign_price_check.csv`) |
| `08_ordering.py` | Basket slopes on USDZAR with the equity channels held fixed (Figure 9, `ordering.csv`, `ordering_baskets.csv`) |
| `09_dual_listed_scatter.py` | Figures 7 and 8 |
| `10_weight_shift.py` | Shift in index weight between categories (`figures/fig_weight_shift.png`) |
| `11_hedge_stack.py` | Figure 10 |
| `12_tables.py` | Tables 1 and 3 (`tables/`) |

## Tables and figures in the paper

| Paper | Script | Output |
|---|---|---|
| Figure 1. Index weight of the Top 40 by category | `04_robustness.py` | `figures/fig_index_composition.png` |
| Figure 2. Two time-conditioned probability profiles | illustration | `figures/fig_probability_profiles.png` |
| Figure 3. Cross-validated R² of the Top 40 as the model is built | `03_figures.py` | `figures/fig_em_ladder_TOP40.png` |
| Figure 4. Share of the Top 40's quarterly movement attributed to each channel | `03_figures.py` | `figures/fig_pf_buildup_TOP40.png` |
| Figure 5. Loading on USDZAR: FTSE/JSE Top 40 and MSCI South Africa | `03_figures.py` | `figures/fig_zar_loading_two.png` |
| Figure 6. The Top 40's loading on USDZAR and its dual-listed weight | `04_robustness.py` | `figures/fig_loading_decomposition.png` |
| Figure 7. Dual-listed basket against USDZAR | `09_dual_listed_scatter.py` | `figures/fig_dual_listed_scatter.png` |
| Figure 8. Dollar-earner basket against USDZAR | `09_dual_listed_scatter.py` | `figures/fig_dollar_earner_scatter.png` |
| Figure 9. Three baskets against USDZAR with the equity channels held fixed | `08_ordering.py` | `figures/fig_ordering_scatter.png` |
| Figure 10. Weight of the Top 40 in dual-listed constituents and dollar-earners | `11_hedge_stack.py` | `figures/fig_hedge_stack.png` |
| Table 1. Descriptive statistics of the weekly log returns | `12_tables.py` | `tables/table_descriptive.csv` |
| Table 2. Classification of Top 40 constituents by where their price is set | rules applied in | `classification_appendix.csv` |
| Table 3. Mean loadings of the Top 40 by period | `12_tables.py` | `tables/table_loadings_by_period.csv` |
| Table 4. The currency results of this study against the prior evidence | compiled from the studies cited | none |
| Table 5. Robustness of the central result | `04_robustness.py` | `results_robustness/` |
| Appendix A. Top 40 constituents by category | | `classification_appendix.csv` |

## Validation against Matthews (2024)

`VALIDATION.md` documents the cross-validation of the flexible-probability routines against the MATLAB toolbox of Matthews (2024), [github.com/nematthews/Meucci-HS-FP-framework-SA-context](https://github.com/nematthews/Meucci-HS-FP-framework-SA-context). To rerun it, clone that repository next to this one, then run `octave validate.m` followed by `python compare.py` from this folder. `height.m` is a one-line Octave shim for MATLAB's `height()`; no function under test is edited.

## Other folders

`results_auto`, `results_cds`, `results_conditioning`, `results_domestic`, the All Share and sub-index output in `results/`, and the matching figures hold output from the specification search that preceded the paper.
