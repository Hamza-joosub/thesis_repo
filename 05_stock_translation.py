"""
05_stock_translation.py — the translation claim tested directly, on the shares it is a claim about.

The claim in Section 4.2 is arithmetic: a dual-listed share's Johannesburg price is its foreign
price converted at the exchange rate, so its weekly rand return is the foreign return plus the
cross-rate return. This script tests that on each pair, then on baskets.

For pair i with foreign return f, cross-rate return x (rand per unit of the foreign currency) and
Johannesburg return j, all weekly log returns:
    j = a + b_f * f + b_x * x + e         translation predicts a = 0, b_f = 1, b_x = 1, high R2
and the one-parameter form
    j = a + b * (f + x) + e               translation predicts b = 1
The second is the sharper test because it imposes the identity. Both are reported.

Then two baskets, equal-weighted across whatever pairs exist in each week:
    foreign-priced basket   the JSE lines of the dual-listed names  -> loading on USDZAR near 1
    domestic basket         banks and retailers                     -> loading near 0 or negative
These bracket the index's loading from above and below, which is what the identity in Section 4.2
requires.

Data: Yahoo Finance via yfinance. Requires: pip install yfinance. Prices are adjusted closes;
Yahoo's dividend adjustment can differ between exchanges for the same company, which adds noise
to the pair test but does not bias it. Weekly returns are Friday to Friday, which absorbs most of
the close-time difference between Johannesburg, London, Sydney and New York; what remains shows
up as R2 below 1, not as a slope away from 1.

Usage
  python 05_stock_translation.py                        all pairs, 1999 on
  python 05_stock_translation.py --start 2008-01-01
  python 05_stock_translation.py --pairs AGL BTI GLN    a subset
Outputs: stock_translation.csv (per-pair results), stock_translation_baskets.csv, and two figures.
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ the pairs
# jse: Johannesburg line (Yahoo, quoted in cents). primary: the home listing and its currency.
# fx: Yahoo cross-rate ticker giving rand per unit of that currency. scale: divisor to get the
# primary quote into major units (London quotes in pence). start: first date the JSE line exists.
PAIRS = {
    'AGL': dict(name='Anglo American',           jse='AGL.JO', primary='AAL.L',   ccy='GBP', fx='GBPZAR=X', scale=100),
    'BTI': dict(name='British American Tobacco', jse='BTI.JO', primary='BATS.L',  ccy='GBP', fx='GBPZAR=X', scale=100),
    'GLN': dict(name='Glencore',                 jse='GLN.JO', primary='GLEN.L',  ccy='GBP', fx='GBPZAR=X', scale=100, start='2013-11-01'),
    'MNP': dict(name='Mondi',                    jse='MNP.JO', primary='MNDI.L',  ccy='GBP', fx='GBPZAR=X', scale=100, start='2007-07-01'),
    'RNI': dict(name='Reinet',                   jse='RNI.JO', primary='REINA.L', ccy='GBP', fx='GBPZAR=X', scale=100, start='2015-12-01'),
    'BHG': dict(name='BHP Group',                jse='BHG.JO', primary='BHP.AX',  ccy='AUD', fx='AUDZAR=X', scale=1,   start='2022-02-01'),
    'S32': dict(name='South32',                  jse='S32.JO', primary='S32.AX',  ccy='AUD', fx='AUDZAR=X', scale=1,   start='2015-05-01'),
    'CFR': dict(name='Richemont',                jse='CFR.JO', primary='CFR.SW',  ccy='CHF', fx='CHFZAR=X', scale=1),
    'PRX': dict(name='Prosus',                   jse='PRX.JO', primary='PRX.AS',  ccy='EUR', fx='EURZAR=X', scale=1,   start='2019-09-01'),
    'ANH': dict(name='AB InBev',                 jse='ANH.JO', primary='ABI.BR',  ccy='EUR', fx='EURZAR=X', scale=1,   start='2016-01-01'),
    'NRP': dict(name='NEPI Rockcastle',          jse='NRP.JO', primary='NRP.AS',  ccy='EUR', fx='EURZAR=X', scale=1,   start='2017-07-01'),
    'ANG': dict(name='AngloGold Ashanti',        jse='ANG.JO', primary='AU',      ccy='USD', fx='USDZAR=X', scale=1,   start='2023-09-01'),
}
# a domestic basket for the bracket: banks and retailers, Johannesburg-primary, rand earnings
DOMESTIC = ['FSR.JO', 'SBK.JO', 'CPI.JO', 'ABG.JO', 'NED.JO', 'SHP.JO', 'WHL.JO', 'TFG.JO', 'MRP.JO', 'CLS.JO']
GLOBAL = {'EM': 'EEM', 'MM': 'XME'}     # proxies for the model's two equity channels, for the basket loadings



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
    import os
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


def fetch(tickers, start, end):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit('pip install yfinance')
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series): df = df.to_frame(tickers[0])
    return df


def weekly_logret(px):
    """Weekly log returns; the panel is already weekly."""
    return np.log(px.resample('W-FRI').last()).diff()


def ols(y, X, names):
    X = np.c_[np.ones(len(X)), X]
    b, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ b; e = y - fit
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # HAC-free standard errors are fine here: the test is on slopes near 1, not on significance
    s2 = (e ** 2).sum() / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return dict(zip(['a'] + names, b)), dict(zip(['a'] + names, se)), r2, len(y)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='1999-01-01'); ap.add_argument('--end', default=None)
    ap.add_argument('--pairs', nargs='*', default=list(PAIRS))
    ap.add_argument('--figdir', default='figures')
    a = ap.parse_args()
    pairs = {k: v for k, v in PAIRS.items() if k in a.pairs}

    print('fetching prices from Yahoo Finance ...')
    tick = sorted({v['jse'] for v in pairs.values()} | {v['primary'] for v in pairs.values()} |
                  {v['fx'] for v in pairs.values()} | set(DOMESTIC) | set(GLOBAL.values()) | {'USDZAR=X'})
    px, SRC = load_prices(tick, a.start, a.end)
    px = px.dropna(axis=1, how='all')
    print(f'   price source: {SRC}')
    missing = [t for t in tick if t not in px.columns]
    if missing: print(f'   not returned by Yahoo: {missing}')
    R = weekly_logret(px)

    # ------------------------------------------------------------ pair by pair
    rows = []
    print(f"\n{'pair':<6}{'name':<26}{'weeks':>6}{'b_f':>7}{'b_x':>7}{'R2 two':>8}{'b (f+x)':>9}{'se':>6}{'R2 one':>8}{'a':>8}")
    for k, v in pairs.items():
        need = [v['jse'], v['primary'], v['fx']]
        if any(t not in R.columns for t in need): print(f'{k:<6}{v["name"]:<26} missing series'); continue
        d = R[need].dropna()
        if v.get('start'): d = d.loc[v['start']:]
        if len(d) < 52: print(f'{k:<6}{v["name"]:<26}{len(d):>6}  too short'); continue
        j = d[v['jse']].values; f = d[v['primary']].values; x = d[v['fx']].values
        # the scale (pence vs pounds) does not affect log returns; it is kept in PAIRS for documentation
        b2, se2, r2_two, n = ols(j, np.c_[f, x], ['f', 'x'])
        b1, se1, r2_one, _ = ols(j, (f + x)[:, None], ['fx'])
        rows.append({'pair': k, 'name': v['name'], 'currency': v['ccy'], 'weeks': n,
                     'b_foreign': b2['f'], 'b_fx': b2['x'], 'r2_two': r2_two,
                     'b_identity': b1['fx'], 'se_identity': se1['fx'], 'r2_identity': r2_one, 'intercept': b1['a'],
                     'first': str(d.index[0].date()), 'last': str(d.index[-1].date())})
        print(f"{k:<6}{v['name']:<26}{n:>6}{b2['f']:>7.2f}{b2['x']:>7.2f}{r2_two:>8.2f}{b1['fx']:>9.2f}{se1['fx']:>6.2f}{r2_one:>8.2f}{b1['a']*100:>7.2f}%")
    P = pd.DataFrame(rows)
    if len(P):
        P.to_csv('stock_translation.csv', index=False)
        print(f"\n  translation predicts b_f = 1, b_x = 1, b(f+x) = 1, intercept 0.")
        print(f"  across pairs: b(f+x) mean {P.b_identity.mean():.2f}, range {P.b_identity.min():.2f}-{P.b_identity.max():.2f}; "
              f"R2 mean {P.r2_identity.mean():.2f}, min {P.r2_identity.min():.2f}")
        wk = (P.b_identity * P.weeks).sum() / P.weeks.sum()
        print(f"  weeks-weighted b(f+x) {wk:.2f}")

    # ------------------------------------------------------------ the two baskets
    jse_lines = [v['jse'] for v in pairs.values() if v['jse'] in R.columns]
    dom = [t for t in DOMESTIC if t in R.columns]
    def basket(cols):
        return R[cols].mean(axis=1, skipna=True).where(R[cols].notna().sum(axis=1) >= 3)
    fp, dm = basket(jse_lines), basket(dom)
    Z = R['USDZAR=X']
    em = R.get(GLOBAL['EM']); mm = R.get(GLOBAL['MM'])
    brows = []
    print(f"\n{'basket':<28}{'weeks':>6}{'loading on USDZAR':>19}{'se':>6}{'with EM and mining':>20}")
    for lab, y in [('foreign-priced, JSE lines', fp), ('domestic, banks and retailers', dm)]:
        d = pd.concat([y, Z], axis=1).dropna(); d.columns = ['y', 'z']
        b, se, r2, n = ols(d.y.values, d[['z']].values, ['z'])
        line = f"{lab:<28}{n:>6}{b['z']:>19.2f}{se['z']:>6.2f}"
        row = {'basket': lab, 'weeks': n, 'loading_usdzar': b['z'], 'se': se['z'], 'r2': r2}
        if em is not None and mm is not None:
            d3 = pd.concat([y, Z, em, mm], axis=1).dropna(); d3.columns = ['y', 'z', 'em', 'mm']
            b3, _, _, _ = ols(d3.y.values, d3[['z', 'em', 'mm']].values, ['z', 'em', 'mm'])
            line += f"{b3['z']:>20.2f}"; row['loading_usdzar_with_channels'] = b3['z']
        print(line); brows.append(row)
    B = pd.DataFrame(brows); B.to_csv('stock_translation_baskets.csv', index=False)
    print('  translation predicts the foreign-priced basket near 1 and the domestic basket near 0 or below.')

    # ------------------------------------------------------------ rolling view: does translation hold through time?
    if len(P):
        os.makedirs(a.figdir, exist_ok=True)
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        UCT = {'navy': '#00243A', 'blue': '#009ADA', 'orange': '#E87722', 'grey': '#B4B4B4', 'mid': '#4F6D7A'}
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for k, v in pairs.items():
            need = [v['jse'], v['primary'], v['fx']]
            if any(t not in R.columns for t in need): continue
            d = R[need].dropna()
            if v.get('start'): d = d.loc[v['start']:]
            if len(d) < 104: continue
            s = d[v['primary']] + d[v['fx']]
            roll = d[v['jse']].rolling(104).cov(s) / s.rolling(104).var()
            ax.plot(roll.index, roll, lw=1.1, label=v['name'], alpha=.9)
        ax.axhline(1, color=UCT['mid'], lw=.8, ls='--'); ax.set_ylim(0, 1.6)
        ax.set_ylabel('rolling 104-week slope of JSE return on (foreign return + cross-rate)')
        ax.set_title('Translation, pair by pair: the slope should sit at one throughout', loc='left', fontsize=10)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc='lower left'); ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(f'{a.figdir}/fig_stock_translation.png', dpi=200); plt.close(fig)
        # bar chart of the identity slopes with standard errors
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        Ps = P.sort_values('b_identity')
        ax.barh(Ps.name, Ps.b_identity, xerr=1.96 * Ps.se_identity, color=UCT['blue'], ecolor=UCT['mid'], capsize=2)
        ax.axvline(1, color=UCT['navy'], lw=1, ls='--'); ax.set_xlim(0, 1.5)
        ax.set_xlabel('slope of JSE weekly return on (foreign return + cross-rate return), with 95% interval')
        ax.set_title('Translation slope by dual-listed pair; one is the prediction', loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.2)
        fig.tight_layout(); fig.savefig(f'{a.figdir}/fig_stock_translation_slopes.png', dpi=200); plt.close(fig)

        # the identity itself: pooled scatter of JSE return on (foreign + cross-rate), slope one
        pool = []
        for k, v in pairs.items():
            need = [v['jse'], v['primary'], v['fx']]
            if any(t not in R.columns for t in need): continue
            d = R[need].dropna()
            if v.get('start'): d = d.loc[v['start']:]
            if k in ('ANH', 'NRP') or len(d) < 52: continue      # broken on Yahoo
            pool.append(pd.DataFrame({'x': d[v['primary']] + d[v['fx']], 'y': d[v['jse']], 'pair': k}))
        if pool:
            Q = pd.concat(pool); Q = Q[(Q.x.abs() < .3) & (Q.y.abs() < .3)]
            bq, _, r2q, nq = ols(Q.y.values, Q[['x']].values, ['x'])
            fig, ax = plt.subplots(figsize=(5.6, 5.4))
            ax.scatter(100 * Q.x, 100 * Q.y, s=4, alpha=.25, color=UCT['blue'], edgecolors='none')
            lim = 25; ax.plot([-lim, lim], [-lim, lim], color=UCT['navy'], lw=1.2, ls='--', label='the identity: slope one')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
            ax.set_xlabel('foreign return + cross-rate return, weekly, %'); ax.set_ylabel('Johannesburg return, weekly, %')
            ax.set_title(f'A dual-listed share\'s rand return is its foreign return plus the exchange rate\n'
                         f'{nq:,} share-weeks across {Q.pair.nunique()} pairs; fitted slope {bq["x"]:.2f}, R² {r2q:.2f}', loc='left', fontsize=9)
            ax.legend(frameon=False, loc='upper left'); ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(f'{a.figdir}/fig_identity_scatter.png', dpi=200); plt.close(fig)
            print(f'   {a.figdir}/fig_identity_scatter.png  (slope {bq["x"]:.3f}, R2 {r2q:.3f}, n {nq})')
        print(f'\nfigures: {a.figdir}/fig_stock_translation.png, {a.figdir}/fig_stock_translation_slopes.png')
