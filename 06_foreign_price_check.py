"""
06_foreign_price_check.py — does the model control for the foreign price of every dual-listed name?

The ratio of the index currency loading to the foreign-priced share should be exactly one only if
each dual-listed name's foreign price is held fixed by the model's equity channels, so that what is
left on USDZAR is the conversion alone. MSCI World Metals and Mining does that for Anglo, BHP and
Glencore because it holds them. It does not hold Richemont, British American Tobacco, Prosus,
AB InBev or Mondi; their foreign prices are controlled only through the emerging-market channel.

For each dual-listed name this script regresses the foreign price return on USDZAR, with and
without the model's channels as controls (proxied here by EEM and XME from Yahoo, and by the actual
MSCI series where the thesis panel is available). A coefficient near zero means the foreign price
does not itself move with the rand once the channels are in, so the JSE line's loading is the
conversion and nothing else. A negative coefficient means the JSE line's loading is one plus that
negative, and the name pulls the index ratio below one in proportion to its weight.

Usage
  python 06_foreign_price_check.py                       Yahoo proxies for the channels
  python 06_foreign_price_check.py --panel data/thesis_data.csv    use the thesis panel's MSCI series
"""
import argparse, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

NAMES = {   # foreign line, its currency's rand cross-rate, whether MSCI World M&M holds it
    'AGL': ('AAL.L',   'GBPZAR=X', True,  'Anglo American'),
    'GLN': ('GLEN.L',  'GBPZAR=X', True,  'Glencore'),
    'BHG': ('BHP.AX',  'AUDZAR=X', True,  'BHP Group'),
    'S32': ('S32.AX',  'AUDZAR=X', True,  'South32'),
    'CFR': ('CFR.SW',  'CHFZAR=X', False, 'Richemont'),
    'BTI': ('BATS.L',  'GBPZAR=X', False, 'British American Tobacco'),
    'PRX': ('PRX.AS',  'EURZAR=X', False, 'Prosus'),
    'MNP': ('MNDI.L',  'GBPZAR=X', False, 'Mondi'),
    'ANG': ('AU',      'USDZAR=X', False, 'AngloGold Ashanti'),
}
# 2023 Top 40 weights of these names, for the weighted contamination estimate (foreign-priced share 46.8)
W2023 = {'AGL': 10.4, 'GLN': 1.3, 'BHG': 1.7, 'S32': 0.0, 'CFR': 21.1, 'BTI': 2.3, 'PRX': 3.7, 'MNP': 2.0, 'ANG': 0.0}


def ols(y, X):
    X = np.c_[np.ones(len(X)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; s2 = (e ** 2).sum() / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return b[1:], se[1:]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2003-10-01'); ap.add_argument('--panel', default=None)
    ap.add_argument('--figdir', default='figures')
    a = ap.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        sys.exit('pip install yfinance')
    tick = sorted({v[0] for v in NAMES.values()} | {v[1] for v in NAMES.values()} | {'USDZAR=X', 'EEM', 'XME'})
    px = yf.download(tick, start=a.start, auto_adjust=True, progress=False)['Close']
    R = np.log(px.resample('W-FRI').last()).diff()
    Z = R['USDZAR=X']
    # channels: the thesis panel's MSCI series if given, else Yahoo proxies
    if a.panel:
        P = pd.read_csv(a.panel, index_col=0, parse_dates=True)
        EM = P['MXEF Index_Log_Diff'].reindex(R.index, method='nearest'); MM = P['MXWO0MM Index_Log_Diff'].reindex(R.index, method='nearest')
        src = 'MSCI EM and MSCI World Metals & Mining from the thesis panel'
    else:
        EM, MM = R['EEM'], R['XME']; src = 'EEM and XME from Yahoo as proxies'
    print(f'channels: {src}\n')
    print(f"{'name':<26}{'in M&M':>7}{'weeks':>6}{'foreign on USDZAR':>19}{'se':>6}{'with EM and mining':>20}{'se':>6}{'implied JSE loading':>21}")
    rows = []
    for k, (fl, fx, inmm, nm) in NAMES.items():
        if fl not in R.columns: print(f'{nm:<26} missing'); continue
        d = pd.concat([R[fl], Z, EM, MM], axis=1).dropna(); d.columns = ['f', 'z', 'em', 'mm']
        if len(d) < 104: continue
        b0, s0 = ols(d.f.values, d[['z']].values)
        b1, s1 = ols(d.f.values, d[['z', 'em', 'mm']].values)
        # the JSE line's loading on USDZAR = 1 (conversion) + the foreign price's own response, controlled
        implied = 1 + b1[0]
        rows.append({'name': nm, 'in_mm': inmm, 'weeks': len(d), 'b_uncond': b0[0], 'b_controlled': b1[0], 'implied_jse_loading': implied, 'w2023': W2023.get(k, 0)})
        print(f"{nm:<26}{'yes' if inmm else 'no':>7}{len(d):>6}{b0[0]:>19.2f}{s0[0]:>6.2f}{b1[0]:>20.2f}{s1[0]:>6.2f}{implied:>21.2f}")
    T = pd.DataFrame(rows); T.to_csv('foreign_price_check.csv', index=False)


    # the figure: a partial regression plot. Frisch-Waugh-Lovell says the coefficient on USDZAR in the
    # full model is the slope of the residual-on-residual scatter once the equity channels are removed
    # from both sides. Raw on the left, partialled on the right.
    JSE = {'AGL': 'AGL.JO', 'GLN': 'GLN.JO', 'BHG': 'BHG.JO', 'S32': 'S32.JO',
           'CFR': 'CFR.JO', 'BTI': 'BTI.JO', 'PRX': 'PRX.JO', 'MNP': 'MNP.JO', 'ANG': 'ANG.JO'}
    jpx = yf.download(list(JSE.values()), start=a.start, auto_adjust=True, progress=False)['Close']
    JR = np.log(jpx.resample('W-FRI').last()).diff()
    raw, par = [], []
    for k, jse in JSE.items():
        if jse not in JR.columns or k not in NAMES: continue
        fl = NAMES[k][0]
        fx = NAMES[k][1]                      # this name's own cross-rate: the identity runs through it, not USDZAR
        if fx not in R.columns: continue
        d = pd.concat([JR[jse], R[fx], EM, MM], axis=1).dropna(); d.columns = ['j', 'z', 'em', 'mm']
        if fl in R.columns:
            first = R[fl].dropna().index.min()
            if pd.notna(first): d = d.loc[first:]
        if len(d) < 104: continue
        raw.append(pd.DataFrame({'x': d.z.values, 'y': d.j.values}))
        X = np.c_[np.ones(len(d)), d[['em', 'mm']].values]
        rj = d.j.values - X @ np.linalg.lstsq(X, d.j.values, rcond=None)[0]
        rz = d.z.values - X @ np.linalg.lstsq(X, d.z.values, rcond=None)[0]
        par.append(pd.DataFrame({'x': rz, 'y': rj}))
    if raw:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
        UCT = {'navy': '#00243A', 'blue': '#009ADA', 'orange': '#E87722', 'mid': '#4F6D7A'}
        os.makedirs(a.figdir, exist_ok=True)
        A = pd.concat(raw); B = pd.concat(par)
        A = A[(A.x.abs() < .12) & (A.y.abs() < .25)]; B = B[(B.x.abs() < .12) & (B.y.abs() < .25)]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharey=True)
        for ax, D, ttl, col, xl in [
                (axes[0], A, '(a) on the exchange rate alone, as the literature measures it', UCT['orange'], "the share's own cross-rate, weekly return, %"),
                (axes[1], B, '(b) with global mining and emerging markets removed from both sides', UCT['blue'], 'cross-rate, residual of the two channels, %')]:
            b = np.polyfit(D.x, D.y, 1); r2 = np.corrcoef(D.x, D.y)[0, 1] ** 2
            ax.scatter(100 * D.x, 100 * D.y, s=3, alpha=.18, color=col, edgecolors='none')
            xs = np.array([-12.0, 12.0])
            ax.plot(xs, 100 * b[1] + b[0] * xs, color=UCT['navy'], lw=1.8, label=f'fitted slope {b[0]:.2f}')
            ax.plot(xs, xs, color=UCT['mid'], lw=1.1, ls='--', label='the identity: slope one')
            ax.set_xlim(-12, 12); ax.set_ylim(-25, 25); ax.set_xlabel(xl)
            ax.set_title(f'{ttl}\nslope {b[0]:.2f}, R\u00b2 {r2:.2f}', loc='left', fontsize=9.5)
            ax.legend(frameon=False, fontsize=8, loc='upper left')
            ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
        axes[0].set_ylabel('Johannesburg weekly return, %')
        fig.suptitle("A dual-listed share's rand response is one, and only the controls reveal it", x=0.01, ha='left', fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f'{a.figdir}/fig_partial_regression.png', dpi=200); plt.close(fig)
        print(f'\n{a.figdir}/fig_partial_regression.png')

    out = T[~T.in_mm]
    if len(out):
        wavg = (out.b_controlled * out.w2023).sum() / max(out.w2023.sum(), 1e-9)
        share = out.w2023.sum() / 46.8
        print(f"\n  names not held by the mining channel: weight-averaged controlled response {wavg:+.2f}")
        print(f"  they are {share:.0%} of the 2023 foreign-priced share, so the index ratio is pulled by about {wavg * share:+.2f}")
        print(f"  a value near zero means the model's channels already control these foreign prices well enough;")
        print(f"  a value of -0.1 or worse means the ratio of one is partly offsetting errors and must be qualified.")
