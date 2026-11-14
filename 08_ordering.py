"""
08_ordering.py — the ordering the paper rests on, shown name by name.

The argument in Section 4.2 needs an ordering, not an exact value: a foreign-priced share responds
to the rand at close to one, a Johannesburg-priced dollar-earner at a fraction of that, and a
domestic company at zero or below. This script estimates each constituent's loading on USDZAR with
the model's two equity channels held fixed, using the same three regressors as the presented model,
and draws them grouped by category. Two panels: the years before the domestic sectors turned, and
the years after.

Where a group's loading sits is a fact about the company's listing and its economics; the ordering
across groups is what makes the index loading track the foreign-priced share and fall when that
share is swapped for dollar-earners at the same weight.

Data: Yahoo Finance via yfinance for the Johannesburg lines; the thesis panel for MSCI EM, MSCI
World Metals & Mining and USDZAR so the controls match the presented model.

Usage
  python 08_ordering.py --panel data/thesis_data.csv
  python 08_ordering.py --panel data/thesis_data.csv --split 2016-01-01
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

GROUPS = {
    'foreign-priced': {'AGL.JO': 'Anglo American', 'GLN.JO': 'Glencore', 'BHG.JO': 'BHP', 'S32.JO': 'South32',
                       'CFR.JO': 'Richemont', 'BTI.JO': 'BAT', 'PRX.JO': 'Prosus', 'MNP.JO': 'Mondi'},
    'SA dollar-earners': {'GFI.JO': 'Gold Fields', 'HAR.JO': 'Harmony', 'IMP.JO': 'Impala', 'SOL.JO': 'Sasol',
                          'SSW.JO': 'Sibanye', 'KIO.JO': 'Kumba', 'EXX.JO': 'Exxaro', 'NPH.JO': 'Northam'},
    'domestic': {'FSR.JO': 'FirstRand', 'SBK.JO': 'Standard Bank', 'CPI.JO': 'Capitec', 'ABG.JO': 'Absa',
                 'NED.JO': 'Nedbank', 'SHP.JO': 'Shoprite', 'WHL.JO': 'Woolworths', 'TFG.JO': 'Foschini',
                 'MRP.JO': 'Mr Price', 'CLS.JO': 'Clicks', 'VOD.JO': 'Vodacom', 'SLM.JO': 'Sanlam'},
}
COL = {'foreign-priced': '#00243A', 'SA dollar-earners': '#E87722', 'domestic': '#009ADA'}
UCT = {'mid': '#4F6D7A', 'grey': '#B4B4B4'}


def ols(y, X):
    X = np.c_[np.ones(len(X)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; s2 = (e ** 2).sum() / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return b[1:], se[1:]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True); ap.add_argument('--start', default='2003-10-01')
    ap.add_argument('--split', default='2016-01-01', help='the domestic sectors turned negative around here')
    ap.add_argument('--figdir', default='figures')
    ap.add_argument('--late-only', action='store_true', help='scatter on the years from --split only')
    ap.add_argument('--mining', default='msci', choices=['msci', 'sp'],
                    help='msci: MSCI World Metals & Mining, which holds Anglo, BHP and Glencore. sp: S&P 500 Metals & Mining, which holds no JSE-listed company, to answer the circularity objection')
    a = ap.parse_args()

# ------------------------------------------------------------------ price source
# bbg_prices_weekly.csv is the Bloomberg pull (PX_LAST, weekly Friday, local currency, 2003 on) parsed
# from dual_listed_prices_pull.xlsx. If it is present it is used; otherwise Yahoo Finance via yfinance.
BBG = 'bbg_prices_weekly.csv'
BBG_MAP = {   # Yahoo ticker -> Bloomberg panel column
    'AGL.JO':'Anglo JSE','AAL.L':'Anglo London','CFR.JO':'Richemont JSE','CFR.SW':'Richemont Zurich',
    'BTI.JO':'BAT JSE','BATS.L':'BAT London','GLN.JO':'Glencore JSE','GLEN.L':'Glencore London',
    'MNP.JO':'Mondi JSE','MNDI.L':'Mondi London','BHG.JO':'BHP JSE','BHP.AX':'BHP Sydney',
    'S32.JO':'South32 JSE','S32.AX':'South32 Sydney','PRX.JO':'Prosus JSE','PRX.AS':'Prosus Amsterdam',
    'ANH.JO':'AB InBev JSE','ABI.BR':'AB InBev Brussels','NRP.JO':'NEPI JSE','NRP.AS':'NEPI Amsterdam',
    'ANG.JO':'AngloGold JSE','AU':'AngloGold New York',
    'GBPZAR=X':'GBP per ZAR','CHFZAR=X':'CHF','AUDZAR=X':'AUD','EURZAR=X':'EUR','USDZAR=X':'USD',
    'GFI.JO':'Gold Fields','HAR.JO':'Harmony','IMP.JO':'Impala','VAL.JO':'Valterra','SSW.JO':'Sibanye','NPH.JO':'Northam',
    'SOL.JO':'Sasol','KIO.JO':'Kumba','EXX.JO':'Exxaro','ARI.JO':'ARM',
    'FSR.JO':'FirstRand','SBK.JO':'Standard Bank','ABG.JO':'Absa','NED.JO':'Nedbank','CPI.JO':'Capitec','SLM.JO':'Sanlam',
    'DSY.JO':'Discovery','OMU.JO':'Old Mutual Ltd','SHP.JO':'Shoprite','WHL.JO':'Woolworths','TFG.JO':'Foschini',
    'MRP.JO':'Mr Price','CLS.JO':'Clicks','TBS.JO':'Tiger Brands','VOD.JO':'Vodacom','GRT.JO':'Growthpoint',
    'BVT.JO':'Bidvest','REM.JO':'Remgro'}


def load_prices(tickers, start, end=None):
    """Weekly closes for the Yahoo-style tickers given, from Bloomberg if the panel exists, else Yahoo."""
    if os.path.exists(BBG):
        P = pd.read_csv(BBG, index_col=0, parse_dates=True)
        cols = {t: BBG_MAP[t] for t in tickers if t in BBG_MAP and BBG_MAP[t] in P.columns}
        out = P[list(cols.values())].copy(); out.columns = list(cols.keys())
        out = out.loc[start:end] if end else out.loc[start:]
        missing = [t for t in tickers if t not in cols]
        if missing: print(f'   not in the Bloomberg panel, skipped: {missing}')
        return out, 'Bloomberg PX_LAST, weekly'
    import yfinance as yf
    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series): df = df.to_frame(list(tickers)[0])
    return df.resample('W-FRI').last(), 'Yahoo Finance adjusted close, weekly'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True); ap.add_argument('--start', default='2003-10-01')
    ap.add_argument('--split', default='2016-01-01', help='the domestic sectors turned negative around here')
    ap.add_argument('--figdir', default='figures')
    ap.add_argument('--late-only', action='store_true', help='scatter on the years from --split only')
    ap.add_argument('--mining', default='msci', choices=['msci', 'sp'],
                    help='msci: MSCI World Metals & Mining, which holds Anglo, BHP and Glencore. sp: S&P 500 Metals & Mining, which holds no JSE-listed company, to answer the circularity objection')
    a = ap.parse_args()
    tick = [t for g in GROUPS.values() for t in g]
    px, SRC = load_prices(tick, a.start)
    print(f'   price source: {SRC}')
    R = np.log(px.resample('W-FRI').last()).diff()
    P = pd.read_csv(a.panel, index_col=0, parse_dates=True)
    mm_col = 'MXWO0MM Index_Log_Diff' if a.mining == 'msci' else 'S5METL Index_Log_Diff'
    ctl = P[['USDZAR Curncy_Log_Diff', 'MXEF Index_Log_Diff', mm_col]].reindex(R.index, method='nearest')
    ctl.columns = ['z', 'em', 'mm']

    # three equal-weight baskets, one per listing category; rolling loadings under the presented model's
    # estimator (ridge, exponential probabilities, tau = 104), so the lines are directly comparable to the index
    from sklearn.linear_model import Ridge
    def wridge(X, y, p, alpha=5.6):
        T = len(p); wm = np.average(X, axis=0, weights=p)
        ws = np.sqrt(np.average((X - wm) ** 2, axis=0, weights=p)); ws[ws == 0] = 1.0
        m = Ridge(alpha=alpha, fit_intercept=True).fit((X - wm) / ws, y, sample_weight=p * T)
        return m.coef_ / ws
    paths = {}; summary = []
    for grp, names in GROUPS.items():
        cols = [t for t in names if t in R.columns]
        basket = R[cols].mean(axis=1, skipna=True).where(R[cols].notna().sum(axis=1) >= 3)
        d = pd.concat([basket, ctl], axis=1).dropna(); d.columns = ['y', 'z', 'em', 'mm']
        X = d[['z', 'em', 'mm']].values; Y = d.y.values; lamb = np.log(2) / 104; out = {}
        for i_ in range(156, len(d)):
            T = i_ + 1; w = np.exp(lamb * np.arange(T)); p = w / w.sum()
            out[d.index[i_]] = wridge(X[:T], Y[:T], p)[0]
        paths[grp] = pd.Series(out)
        s_, e_ = paths[grp].iloc[:52].mean(), paths[grp].iloc[-52:].mean()
        summary.append({'group': grp, 'names': len(cols), 'first': str(paths[grp].index[0].date()), 'start': s_, 'end': e_})
        print(f"{grp:<20} {len(cols):>2} names   loading {s_:+.2f} -> {e_:+.2f}   from {paths[grp].index[0].date()}")
    pd.DataFrame(summary).to_csv('ordering_baskets.csv', index=False)

    # the scatter: each basket's return against USDZAR, both with the equity channels partialled out
    # (Frisch-Waugh-Lovell), so the slope of each panel is the basket's loading in the presented model
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(a.figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), sharey=True)
    for ax, (grp, names) in zip(axes, GROUPS.items()):
        cols = [t for t in names if t in R.columns]
        basket = R[cols].mean(axis=1, skipna=True).where(R[cols].notna().sum(axis=1) >= 3)
        d = pd.concat([basket, ctl], axis=1).dropna(); d.columns = ['y', 'z', 'em', 'mm']
        d = d.loc[a.split:] if a.late_only else d
        X = np.c_[np.ones(len(d)), d[['em', 'mm']].values]
        ry = d.y.values - X @ np.linalg.lstsq(X, d.y.values, rcond=None)[0]
        rz = d.z.values - X @ np.linalg.lstsq(X, d.z.values, rcond=None)[0]
        keep = (np.abs(rz) < .08) & (np.abs(ry) < .15); ry, rz = ry[keep], rz[keep]
        b = np.polyfit(rz, ry, 1); r2 = np.corrcoef(rz, ry)[0, 1] ** 2
        ax.scatter(100 * rz, 100 * ry, s=5, alpha=.3, color=COL[grp], edgecolors='none')
        xs = np.array([-8.0, 8.0])
        ax.plot(xs, 100 * b[1] + b[0] * xs, color='#00243A', lw=2, label=f'fitted slope {b[0]:+.2f}')
        ax.plot(xs, xs, color=UCT['mid'], lw=1.1, ls='--', label='the identity: slope one')
        ax.axhline(0, color=UCT['grey'], lw=.6); ax.axvline(0, color=UCT['grey'], lw=.6)
        ax.set_xlim(-8, 8); ax.set_ylim(-15, 15)
        ax.set_title(f'{grp}   slope {b[0]:+.2f}, R\u00b2 {r2:.2f}', loc='left', fontsize=9.5)
        ax.set_xlabel('USDZAR weekly return, %\nwith the equity channels removed')
        ax.legend(frameon=False, fontsize=7.5, loc='upper left'); ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
    axes[0].set_ylabel('basket weekly return, %, equity channels removed')
    per = f'from {a.split[:4]}' if a.late_only else 'full sample'
    mlab = 'MSCI World Metals & Mining' if a.mining == 'msci' else 'S&P 500 Metals & Mining, no JSE constituents'
    fig.suptitle(f'Response to the rand by where the price is set, equal-weight baskets of Top 40 constituents, {per}; mining control {mlab}', x=0.01, ha='left', fontsize=10)
    tag = '' if a.mining == 'msci' else '_spmining'
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f'{a.figdir}/fig_ordering_scatter{tag}.png', dpi=200); plt.close(fig)
    print(f'{a.figdir}/fig_ordering_scatter{tag}.png')

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for grp, sr in paths.items():
        sm = sr.rolling(13, min_periods=1).mean()
        ax.plot(sm.index, sm, color=COL[grp], lw=1.9, label=grp)
        ax.annotate(f'{grp}  {sr.iloc[-52:].mean():+.2f}', xy=(sm.index[-1], sm.iloc[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8, color=COL[grp], va='center')
    ax.axhline(1, color='#00243A', lw=.9, ls='--'); ax.axhline(0, color=UCT['mid'], lw=.8)
    ax.annotate('the identity: one', xy=(ax.get_xlim()[0], 1.0), xytext=(4, 4), textcoords='offset points', fontsize=7.5, color='#00243A')
    ax.set_ylabel('loading on USDZAR, equity channels held fixed')
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 800)
    ax.set_title('Three baskets of Top 40 constituents, by where their price is set', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower left'); ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(f'{a.figdir}/fig_ordering{tag}.png', dpi=200); plt.close(fig)
    print(f'{a.figdir}/fig_ordering{tag}.png')
