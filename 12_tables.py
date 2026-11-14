"""12_tables.py: the two tables the sample papers all carry and the draft lacks.

Table A. Descriptive statistics of the weekly log returns, with an ADF unit-root test and the correlation matrix.
Table B. The presented model's loadings by period, with the bootstrap standard error on the USDZAR loading.

Usage:
  python 12_tables.py
  python 12_tables.py --panel transformed_data.csv --betas results/tau_104/TOP40/em_ladder_2/betas_raw.csv --boot results_robustness/9_bootstrap.csv
Writes tables/table_descriptive.csv, tables/table_correlations.csv, tables/table_loadings_by_period.csv and prints all three.
"""
import argparse, os, numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller

ap = argparse.ArgumentParser()
ap.add_argument('--panel', default='transformed_data.csv')
ap.add_argument('--betas', default='results/tau_104/TOP40/em_ladder_2/betas_raw.csv')
ap.add_argument('--boot', default='results_robustness/9_bootstrap.csv')
ap.add_argument('--start', default='1999-01-01')
a = ap.parse_args(); os.makedirs('tables', exist_ok=True)

SER = {'TOP40 Index_Log_Diff': 'FTSE/JSE Top 40', 'MXEF Index_Log_Diff': 'MSCI Emerging Markets',
       'MXWO0MM Index_Log_Diff': 'MSCI World Metals & Mining', 'USDZAR Curncy_Log_Diff': 'USDZAR'}
d = pd.read_csv(a.panel, index_col=0, parse_dates=True).loc[a.start:, list(SER)].dropna()

# Table A: descriptive statistics, weekly log returns in percent
rows = []
for c, nm in SER.items():
    x = 100 * d[c]
    adf = adfuller(d[c], autolag='AIC')
    rows.append({'Series': nm, 'N': len(x), 'Mean': x.mean(), 'Std dev': x.std(), 'Min': x.min(), 'Max': x.max(),
                 'Skewness': x.skew(), 'Excess kurtosis': x.kurt(), 'ADF stat': adf[0], 'ADF p': adf[1]})
A = pd.DataFrame(rows).set_index('Series')
A.round(3).to_csv('tables/table_descriptive.csv')
C = d.rename(columns=SER).corr(); C.round(2).to_csv('tables/table_correlations.csv')
print(f'Table A. Weekly log returns, percent, {d.index.min().date()} to {d.index.max().date()}')
print(A.round(3).to_string()); print('\nCorrelations'); print(C.round(2).to_string())

# Table B: loadings by period
b = pd.read_csv(a.betas, index_col=0, parse_dates=True)
def pick(key): return [c for c in b.columns if key in c.upper()][0]
cols = {}
for key, nm in [('MXEF', 'EM'), ('MM', 'Mining'), ('ZAR', 'USDZAR')]:
    try: cols[pick(key)] = nm
    except IndexError: pass
if len(cols) < 3:                       # unnamed columns: the presented spec's order is EM, USDZAR, mining
    cols = dict(zip(b.columns[:3], ['EM', 'USDZAR', 'Mining']))
b = b[list(cols)].rename(columns=cols)[['EM', 'Mining', 'USDZAR']]
periods = [(2003, 2007), (2008, 2011), (2012, 2015), (2016, 2019), (2020, 2023), (2024, 2026)]
se = None
if os.path.exists(a.boot):
    bt = pd.read_csv(a.boot); se = bt.set_index('year')['se'] if 'se' in bt.columns else None
rows = []
for lo, hi in periods + [(2003, 2026)]:
    g = b[(b.index.year >= lo) & (b.index.year <= hi)]
    r = {'Period': f'{lo}–{hi}' if (lo, hi) != (2003, 2026) else 'Full sample', 'Weeks': len(g)}
    for c in ['EM', 'Mining', 'USDZAR']: r[c] = g[c].mean()
    if se is not None: r['SE USDZAR (bootstrap, mean)'] = se.loc[(se.index >= lo) & (se.index <= hi)].mean()
    rows.append(r)
B = pd.DataFrame(rows).set_index('Period')
B.round(3).to_csv('tables/table_loadings_by_period.csv')
print('\nTable B. Mean loadings by period, presented model (tau = 104, alpha = 5.6)')
print(B.round(3).to_string())
