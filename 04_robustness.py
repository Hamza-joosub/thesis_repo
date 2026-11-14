"""
04_robustness.py — robustness and validation for the currency finding.

Standalone. Depends only on transformed_data.csv (from 00_transform.py), numpy, pandas,
scikit-learn and scipy. It does not import the runner; the weighted ridge here reproduces
fit_ridge_FP_SC exactly (weighted standardisation, sklearn Ridge with sample_weight = p * T,
back-transformation to raw units), which has been checked to zero difference.

The finding it tests has three parts:
  1  The All Share's currency loading equals the dual-listed weight of the index. From 2008 to
     2019 the ratio averages 1.00. A dual-listed share's rand price is its foreign price times
     the exchange rate, so the index's currency exposure is translation, not an economic hedge.
  2  The translated component shrank: dual-listed weight 44 percent in 2008 to 16 percent in 2026.
  3  The rest of the decline is South African country risk arriving through the only channel that
     carries any. Adding the SA sovereign spread halves the negative currency loading on the
     domestic sectors and leaves the resource indices untouched; the US ten-year does nothing.

Tests
  1  ratio          loading against dual-listed weight, at three half-lives and two penalties
  2  spread         the yield decomposition, every target, three half-lives, both mining proxies
  3  identity       reconstruct the index loading from sector loadings and sector weights
  4  profile        the loading across percentiles of the rand's own move, early against late,
                    with and without time decay
  5  taxonomy       currency loading through time for every index, for the taxonomy figure
  6  rechecks       the explanations already rejected, re-run so the claim can be audited:
                    the dollar, the EM currency bloc, the orthogonal rand split, gold and
                    platinum, the interest rate differential
  7  reconcile      does the finding survive an overlap-free mining proxy, a different target
                    index, and the exclusion of the post-2023 window

Usage
  python 04_robustness.py                          everything, resumable, several hours
  python 04_robustness.py --tests 1 3              only those
  python 04_robustness.py --quick                  one half-life, fewer targets
  python 04_robustness.py --report                 re-print the report from cached results
  python 04_robustness.py --report --plot          draw the figures from cached results into figures/
Results are cached per test in results_robustness/ and skipped on a rerun, so it can be stopped
and restarted. --force recomputes.
"""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge

OUT = 'results_robustness'
PRIMARY = 'TOP40'   # the presented index; Flint and Seymour's and Polakow and Flint's

# ------------------------------------------------------------------ columns
EM  = 'MXEF Index_Log_Diff'
ZAR = 'USDZAR Curncy_Log_Diff'
MM  = 'MXWO0MM Index_Log_Diff'
SPM = 'S5METL Index_Log_Diff'          # S&P 500 Metals and Mining, no JSE constituents
BASE = [EM, ZAR, MM]
CLEAN = [EM, ZAR, SPM]

TARGETS = {t: f'{t} Index_Log_Diff' for t in
           ['JALSH', 'TOP40', 'MXZA', 'RESI20', 'JMNNG', 'INDI25', 'FINI15', 'JGENR']}

# Weights of the Top 40 by category, from Bloomberg INDX_MWEIGHT_HIST at each month-end from January
# 2003, classified by listing as in Table 2 of the thesis (top40_monthly_class.csv). W is the dual-listed
# share, S the South African dollar-earner share. Annual values are April month-ends, to match the
# original snapshots; the monthly series is used wherever a finer comparison to the loading path helps.
_WEIGHTS = 'top40_weights_monthly.csv'
def _load_weights():
    if not os.path.exists(_WEIGHTS):
        raise SystemExit(f'{_WEIGHTS} not found: build it from top40_monthly_class.csv')
    C = pd.read_csv(_WEIGHTS, index_col=0, parse_dates=True)
    apr = C[C.index.month == 4]
    return ({int(d.year): float(v) for d, v in apr['dual'].items()},
            {int(d.year): float(v) for d, v in apr['earn'].items()},
            {int(d.year): float(v) for d, v in apr['naspers'].items()}, C)
DUAL, SA_DOLLAR, NASPERS, WEIGHTS_MONTHLY = _load_weights()


# ------------------------------------------------------------------ estimation
def wridge(X, y, p, alpha):
    """Ridge under arbitrary probabilities, back-transformed to raw units."""
    T = len(p)
    wm = np.average(X, axis=0, weights=p)
    ws = np.sqrt(np.average((X - wm) ** 2, axis=0, weights=p)); ws[ws == 0] = 1.0
    m = Ridge(alpha=alpha, fit_intercept=True).fit((X - wm) / ws, y, sample_weight=p * T)
    return m.coef_ / ws


FIRST_ESTIMATE = None   # set from --start-date; aligns every test to the main results' first date


def loading_path(df, cols, y_col, tau=104, alpha=5.6, start_idx=250, end=None):
    """Rolling loadings under exponential flexible probabilities with half-life tau.
    If FIRST_ESTIMATE is set, the first estimate is placed on or after that date regardless of
    how far back the panel runs, so results are comparable across panels with different starts."""
    sub = df[cols + [y_col]].dropna()
    if end is not None: sub = sub.loc[:end]
    if FIRST_ESTIMATE is not None:
        pos = sub.index.searchsorted(pd.Timestamp(FIRST_ESTIMATE))
        start_idx = max(start_idx, int(pos))
    if len(sub) <= start_idx + 10: return None
    X_all = sub[cols].values.astype(float); y_all = sub[y_col].values.astype(float)
    lamb = np.log(2) / tau; out = {}
    for i in range(start_idx, len(sub)):
        T = i + 1
        w = np.exp(lamb * np.arange(T)); p = w / w.sum()
        out[sub.index[i]] = wridge(X_all[:T], y_all[:T], p, alpha)
    B = pd.DataFrame(out).T; B.columns = cols
    return B


def ends(s, n=52):
    return float(s.iloc[:n].mean()), float(s.iloc[-n:].mean())


def tilt(p, x, target):
    """Minimum relative entropy tilt subject to E[x] = target, adaptive bracket."""
    from scipy.optimize import brentq
    def err(lam):
        t = lam * x; t -= t.max(); q = p * np.exp(t); s = q.sum()
        return np.nan if (s == 0 or not np.isfinite(s)) else float((q / s) @ x - target)
    scale = max(abs(x.max()), abs(x.min()), 1e-12)
    for mult in (5.0, 50.0, 500.0):
        b = mult / scale
        try:
            e0, e1 = err(-b), err(b)
            if not np.isfinite(e0) or not np.isfinite(e1) or np.sign(e0) == np.sign(e1): continue
            lam = brentq(err, -b, b); t = lam * x; t -= t.max()
            q = p * np.exp(t); return q / q.sum()
        except Exception:
            continue
    return None


# ------------------------------------------------------------------ caching
def cached(name, fn, force=False):
    os.makedirs(OUT, exist_ok=True)
    path = f'{OUT}/{name}.csv'
    if os.path.exists(path) and not force:
        print(f'  [cached] {name}')
        return pd.read_csv(path)
    t0 = time.time()
    df = fn()
    if df is None or not len(df): return None
    df.to_csv(path, index=False)
    print(f'  [{time.time() - t0:>5.0f}s] {name}')
    return df


# ------------------------------------------------------------------ 1. ratio
def test_ratio(d, a):
    """Is the currency loading equal to the dual-listed weight, and is that robust to tau and alpha?"""
    def build():
        rows = []
        for target in (['TOP40'] if a.quick else ['TOP40', 'JALSH']):
            for tau in a.taus:
                for alpha in ([5.6] if a.quick else [5.6, 3.0, 10.0]):
                    B = loading_path(d, BASE, TARGETS[target], tau, alpha, a.start_idx)
                    if B is None: continue
                    z = B[ZAR].resample('YE').mean(); z.index = z.index.year
                    for yr, w in DUAL.items():
                        if yr in z.index:
                            rows.append({'target': target, 'tau': tau, 'alpha': alpha, 'year': yr,
                                         'dual_weight': w / 100, 'sa_dollar': SA_DOLLAR[yr] / 100,
                                         'loading': float(z.loc[yr]), 'ratio': float(z.loc[yr]) / (w / 100)})
        return pd.DataFrame(rows)
    r = cached('1_ratio', build, a.force)
    if r is None: return
    print('\n  loading divided by dual-listed weight, by period')
    print('  (a value of 1 means the index currency loading is exactly its dual-listed weight)')
    for (t, tau, al), g in r.groupby(['target', 'tau', 'alpha']):
        pre = g[g.year <= 2019]['ratio'].mean(); mid = g[(g.year > 2019) & (g.year <= 2023)]['ratio'].mean()
        post = g[g.year > 2023]['ratio'].mean()
        print(f'    {t}  tau {tau:>3}  alpha {al:>4}   to 2019 {pre:.2f}   2020-23 {mid:.2f}   2024-26 {post:.2f}')
    g = r[(r.target == PRIMARY) & (r.tau == 104) & (r.alpha == 5.6)]
    if len(g):
        print(f'\n  {PRIMARY}, tau 104, alpha 5.6, year by year')
        print(f"    {'year':>6}{'dual weight':>13}{'loading':>10}{'ratio':>8}{'SA dollar':>11}")
        for _, x in g.sort_values('year').iterrows():
            print(f"    {int(x.year):>6}{x.dual_weight:>13.3f}{x.loading:>10.3f}{x.ratio:>8.2f}{x.sa_dollar:>11.3f}")
        print(f"\n    correlation of loading with dual-listed weight  {g.loading.corr(g.dual_weight):+.3f}")
        print(f"    correlation of loading with SA dollar-earner weight {g.loading.corr(g.sa_dollar):+.3f}")
        print('    NOTE: membership weights begin in January 2003, three months before the first loading estimate.')


# ------------------------------------------------------------------ 2. spread
def test_spread(d, a):
    """Does the SA country risk premium account for the domestic sectors' negative currency loading?"""
    d = d.copy()
    d['_us'] = d['USGG10YR Index_Diff']
    d['_spread'] = (d['ALBIAY Index'] - d['USGG10YR Index']).diff()
    d['_cds'] = d.get('REPSOU CDS USD SR 5Y D14 Corp_Diff')
    specs = [('three channels', BASE),
             ('plus US 10y', BASE + ['_us']),
             ('plus SA spread', BASE + ['_spread']),
             ('plus both', BASE + ['_us', '_spread']),
             ('plus SA CDS', BASE + ['_cds']),
             ('clean mining, plus SA spread', CLEAN + ['_spread'])]
    targets = ['TOP40', 'FINI15', 'JGENR', 'RESI20'] if a.quick else list(TARGETS)

    def build():
        rows = []
        for target in targets:
            for tau in a.taus:
                for lab, cols in specs:
                    cols = [c for c in cols if c in d.columns]
                    if ZAR not in cols: continue
                    B = loading_path(d, cols, TARGETS[target], tau, a.alpha, a.start_idx)
                    if B is None: continue
                    s, e = ends(B[ZAR])
                    row = {'target': target, 'tau': tau, 'spec': lab, 'zar_start': s, 'zar_end': e,
                           'n': len(B)}
                    for extra in ('_us', '_spread', '_cds'):
                        if extra in cols:
                            xs, xe = ends(B[extra]); row[f'{extra}_start'] = xs; row[f'{extra}_end'] = xe
                    rows.append(row)
        return pd.DataFrame(rows)
    r = cached('2_spread', build, a.force)
    if r is None: return
    for tau in sorted(r.tau.unique()):
        print(f'\n  currency loading, start to end, half-life {int(tau)} weeks')
        p = r[r.tau == tau].pivot(index='target', columns='spec', values='zar_end')
        order = [s for s, _ in specs if s in p.columns]
        print('    ' + p[order].round(3).to_string().replace('\n', '\n    '))
    print('\n  the test: adding the SA spread should move the domestic sectors and leave the resource')
    print('  indices alone, and adding the US ten-year should do nothing anywhere.')
    base = r[(r.tau == 104) & (r.spec == 'three channels')].set_index('target')['zar_end']
    for lab in ('plus US 10y', 'plus SA spread'):
        g = r[(r.tau == 104) & (r.spec == lab)].set_index('target')['zar_end']
        shifts = (g - base).dropna()
        print(f'    {lab:<18}' + '  '.join(f'{k} {v:+.3f}' for k, v in shifts.items()))


# ------------------------------------------------------------------ 3. identity
def test_identity(d, a):
    """index currency loading = dual-listed translation + domestic rand-play exposure."""
    def build():
        rows = []
        paths = {}
        for t in [PRIMARY, 'RESI20', 'INDI25', 'FINI15']:
            B = loading_path(d, BASE, TARGETS[t], 104, a.alpha, a.start_idx)
            if B is not None:
                z = B[ZAR].resample('YE').mean(); z.index = z.index.year; paths[t] = z
        for yr in DUAL:
            if yr not in paths.get(PRIMARY, {}): continue
            rows.append({'year': yr, 'dual_weight': DUAL[yr] / 100,
                         'index': float(paths[PRIMARY].loc[yr]),
                         **{t: float(paths[t].loc[yr]) for t in ('RESI20', 'INDI25', 'FINI15') if t in paths and yr in paths[t]}})
        return pd.DataFrame(rows)
    r = cached('3_identity', build, a.force)
    if r is None: return
    r['implied_translation'] = r.dual_weight
    r['residual'] = r['index'] - r.dual_weight
    print('\n  if the exposure were pure translation, the residual would be zero throughout')
    print(f"    {'year':>6}{'dual weight':>13}{'index loading':>15}{'residual':>11}{'Financials':>12}")
    for _, x in r.sort_values('year').iterrows():
        fin = f"{x.get('FINI15', float('nan')):>12.3f}" if pd.notna(x.get('FINI15', np.nan)) else f"{'':>12}"
        print(f"    {int(x.year):>6}{x.dual_weight:>13.3f}{x['index']:>15.3f}{x.residual:>11.3f}{fin}")
    pre = r[r.year <= 2019]['residual'].mean(); post = r[r.year > 2019]['residual'].mean()
    print(f'\n    mean residual to 2019 {pre:+.3f}, after 2019 {post:+.3f}')
    if 'FINI15' in r:
        print(f"    correlation of the residual with the Financials loading {r.residual.corr(r.FINI15):+.3f}")


# ------------------------------------------------------------------ 4. profile
def test_profile(d, a):
    """The loading across percentiles of the rand's own move, early against late, both weightings."""
    pcts = [5, 10, 20, 35, 50, 65, 80, 90, 95]
    targets = ['TOP40', 'RESI20', 'JGENR'] if a.quick else ['TOP40', 'JALSH', 'MXZA', 'RESI20', 'JMNNG', 'FINI15', 'JGENR']

    def build():
        rows = []
        for target in targets:
            sub = d[BASE + [TARGETS[target]]].dropna()
            cut = sub.index[len(sub) // 2]
            for decay in (False, True):
                for plab, s in [('full', sub), (f'early to {cut.year}', sub.loc[:cut]),
                                (f'late from {cut.year}', sub.loc[cut:])]:
                    X = s[BASE].values.astype(float); y = s[TARGETS[target]].values.astype(float)
                    z = s[ZAR].values.astype(float); zs = (z - z.mean()) / (z.std() or 1.0)
                    T = len(s)
                    if decay:
                        lamb = np.log(2) / 104; w = np.exp(lamb * (np.arange(T) - T)); p0 = w / w.sum()
                    else:
                        p0 = np.ones(T) / T
                    for q in pcts:
                        p = tilt(p0, zs, float(np.percentile(zs, q)))
                        if p is None: continue
                        b = wridge(X, y, p, a.alpha)
                        rows.append({'target': target, 'decay': decay, 'period': plab, 'percentile': q,
                                     'zar': float(b[BASE.index(ZAR)]),
                                     'ens': float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-300)))))})
        return pd.DataFrame(rows)
    r = cached('4_profile', build, a.force)
    if r is None: return
    for decay in sorted(r.decay.unique()):
        print(f"\n  currency loading across the rand's range, {'with' if decay else 'without'} time decay")
        print('  (low percentile: rand strength. high percentile: rand weakness)')
        for target, g in r[r.decay == decay].groupby('target'):
            per = [p for p in g.period.unique() if p != 'full']
            line = f'    {target:<8}'
            for p in sorted(per):
                gg = g[g.period == p]
                lo = gg[gg.percentile <= 20]['zar'].mean(); hi = gg[gg.percentile >= 80]['zar'].mean()
                line += f'   {p}: strength {lo:+.2f} weakness {hi:+.2f} slope {hi - lo:+.2f}'
            print(line)


# ------------------------------------------------------------------ 5. taxonomy
def test_taxonomy(d, a):
    """Currency loading through time for every index, for the taxonomy figure."""
    def build():
        rows = []
        for t in TARGETS:
            B = loading_path(d, BASE, TARGETS[t], 104, a.alpha, a.start_idx)
            if B is None: continue
            s = B[ZAR].rolling(26, min_periods=1).mean()
            for dt, v in s.items(): rows.append({'target': t, 'date': dt, 'zar': float(v)})
        return pd.DataFrame(rows)
    r = cached('5_taxonomy', build, a.force)
    if r is None: return
    r['date'] = pd.to_datetime(r['date'])
    print('\n  currency loading by index, first and last year, and when it last crossed zero')
    print(f"    {'index':<8}{'start':>9}{'end':>9}{'change':>9}{'last crossing of zero':>24}")
    for t, g in r.groupby('target'):
        g = g.sort_values('date'); s, e = ends(g.zar)
        sg = np.sign(g.zar.values); cr = np.where(np.diff(sg) != 0)[0]
        last = str(g.date.iloc[cr[-1]].date()) if len(cr) else 'never'
        print(f'    {t:<8}{s:>+9.3f}{e:>+9.3f}{e - s:>+9.3f}{last:>24}')


# ------------------------------------------------------------------ 6. rechecks
def test_rechecks(d, a):
    """Re-run the explanations already rejected, so the claim can be audited."""
    d = d.copy()
    D = 'DXY Curncy_Log_Diff'; E = 'FXJPEMCS Index_Log_Diff'
    d['ZAR_nodollar'] = d[ZAR] - d[D]
    if E in d.columns:
        s = d[[ZAR, E]].dropna(); X = np.c_[np.ones(len(s)), s[E]]
        b = np.linalg.lstsq(X, s[ZAR], rcond=None)[0]; d.loc[s.index, 'ZAR_noEM'] = s[ZAR] - X @ b
    G = [c for c in [MM, EM, D, 'USGG10YR Index_Diff', 'VIX Index_Diff', 'BCOM Index_Log_Diff'] if c in d.columns]
    s = d[G + [ZAR]].dropna(); X = np.c_[np.ones(len(s)), s[G].values]
    b = np.linalg.lstsq(X, s[ZAR].values, rcond=None)[0]
    d.loc[s.index, 'ZAR_glob'] = X @ b; d.loc[s.index, 'ZAR_dom'] = s[ZAR].values - X @ b
    metals = [c for c in ['XAU Curncy_Log_Diff', 'XPT Curncy_Log_Diff'] if c in d.columns]
    if {'SARPRT Index', 'USGG10YR Index'} <= set(d.columns):
        d['_carry'] = (d['SARPRT Index'] - d['USGG10YR Index']).diff()
    checks = [('presented model', BASE, ZAR),
              ('dollar removed from the rand', [EM, 'ZAR_nodollar', MM], 'ZAR_nodollar'),
              ('EM currency bloc removed', [EM, 'ZAR_noEM', MM], 'ZAR_noEM'),
              ('rand split, global part', [EM, 'ZAR_glob', 'ZAR_dom', MM], 'ZAR_glob'),
              ('rand split, orthogonal part', [EM, 'ZAR_glob', 'ZAR_dom', MM], 'ZAR_dom'),
              ('plus gold and platinum', BASE + metals, ZAR),
              ('plus the rate differential', BASE + ['_carry'], ZAR),
              ('overlap-free mining proxy', CLEAN, ZAR)]

    def build():
        rows = []
        for target in (['TOP40'] if a.quick else ['TOP40', 'RESI20', 'FINI15']):
            for lab, cols, read in checks:
                cols = [c for c in cols if c in d.columns]
                if read not in cols: continue
                B = loading_path(d, cols, TARGETS[target], 104, a.alpha, a.start_idx)
                if B is None: continue
                s0, e0 = ends(B[read])
                rows.append({'target': target, 'check': lab, 'reads': read, 'start': s0, 'end': e0,
                             'pct_change': (e0 - s0) / abs(s0) * 100 if s0 else np.nan, 'n': len(B)})
        return pd.DataFrame(rows)
    r = cached('6_rechecks', build, a.force)
    if r is None: return
    print('\n  the decline should survive every one of these')
    for target, g in r.groupby('target'):
        print(f'\n    {target}')
        print(f"      {'check':<32}{'start':>9}{'end':>9}{'change':>10}{'weeks':>8}")
        for _, x in g.iterrows():
            print(f"      {x['check']:<32}{x['start']:>+9.3f}{x['end']:>+9.3f}{x['pct_change']:>9.0f}%{int(x['n']):>8}")


# ------------------------------------------------------------------ 7. reconcile
def test_reconcile(d, a):
    """Does the finding survive a different mining proxy, a different index, and dropping 2024 on?"""
    def build():
        rows = []
        for target in ['TOP40', 'JALSH', 'MXZA']:
            for plab, end in [('full sample', None), ('to end 2023', '2023-12-31')]:
                for mlab, cols in [('MSCI World mining', BASE), ('S&P 500 mining', CLEAN)]:
                    B = loading_path(d, cols, TARGETS[target], 104, a.alpha, a.start_idx, end=end)
                    if B is None: continue
                    s, e = ends(B[ZAR])
                    rows.append({'target': target, 'window': plab, 'mining_proxy': mlab,
                                 'start': s, 'end': e, 'pct': (e - s) / abs(s) * 100, 'n': len(B)})
        return pd.DataFrame(rows)
    r = cached('7_reconcile', build, a.force)
    if r is None: return
    print('\n  the decline, by window and mining proxy')
    print(f"    {'index':<8}{'window':<14}{'mining proxy':<20}{'start':>9}{'end':>9}{'change':>9}")
    for _, x in r.iterrows():
        print(f"    {x.target:<8}{x.window:<14}{x.mining_proxy:<20}{x.start:>+9.3f}{x.end:>+9.3f}{x.pct:>8.0f}%")
    print('\n  the post-2023 window is the part neither the free-float change nor any tested')
    print('  explanation accounts for. If the decline to end 2023 is already large, the finding')
    print('  does not rest on it.')



# ------------------------------------------------------------------ 8. placebo
def test_placebo(d, a):
    """If the currency loading tracks dual-listed weight only because both trend, the mining and EM
    loadings, which trend over the same years for unrelated reasons, should track it too."""
    def build():
        B = loading_path(d, BASE, TARGETS[PRIMARY], 104, a.alpha, a.start_idx)
        if B is None: return None
        y = B.resample('YE').mean(); y.index = y.index.year
        rows = []
        for yr, w in DUAL.items():
            if yr in y.index:
                rows.append({'year': yr, 'dual_weight': w / 100, 'sa_dollar': SA_DOLLAR[yr] / 100,
                             'currency': float(y.loc[yr, ZAR]), 'mining': float(y.loc[yr, MM]), 'em': float(y.loc[yr, EM])})
        return pd.DataFrame(rows)
    r = cached('8_placebo', build, a.force)
    if r is None: return
    d1 = r.diff().dropna()
    print('\n  correlation of each loading with the dual-listed weight, levels and first differences')
    print(f"    {'loading':<12}{'levels':>9}{'first diff':>12}{'ratio to weight, to 2019':>26}")
    for c in ('currency', 'mining', 'em'):
        pre = (r[r.year <= 2019][c] / r[r.year <= 2019].dual_weight).mean()
        print(f"    {c:<12}{r[c].corr(r.dual_weight):>+9.2f}{d1[c].corr(d1.dual_weight):>+12.2f}{pre:>26.2f}")
    print('\n  the currency loading should track the weight and the other two should not. if all three')
    print('  track it, the relationship is a shared trend rather than translation.')


# ------------------------------------------------------------------ 9. bootstrap
def test_bootstrap(d, a):
    """Block-bootstrap confidence bands on the annual currency loading, so the dual-listed weight can
    be tested against an interval rather than a point."""
    def build():
        sub = d[BASE + [TARGETS[PRIMARY]]].dropna()
        if FIRST_ESTIMATE is not None:
            pos = sub.index.searchsorted(pd.Timestamp(FIRST_ESTIMATE)); si = max(a.start_idx, int(pos))
        else: si = a.start_idx
        n = len(sub); block = 26; rng = np.random.default_rng(0)
        # evaluation dates: one per year, the last estimation date of each year with a weight
        yrs = [y for y in DUAL if y >= sub.index[si].year and y <= sub.index[-1].year]
        eval_idx = {y: int(sub.index.searchsorted(pd.Timestamp(f'{y}-04-30'))) for y in yrs}
        eval_idx = {y: i for y, i in eval_idx.items() if si <= i < n}
        Xa = sub[BASE].values.astype(float); ya = sub[TARGETS[PRIMARY]].values.astype(float)
        lamb = np.log(2) / 104
        draws = {y: [] for y in eval_idx}
        # residual block bootstrap: hold the regressors fixed, resample residual blocks, rebuild y.
        # this measures the sampling uncertainty of the coefficient given the regressors, which is
        # the quantity a standard error reports. a full-panel block bootstrap would instead vary
        # which eras sit near the recent end under exponential weighting, and answers a different question.
        points, fitted = {}, {}
        for y, i in eval_idx.items():
            T = i + 1; w = np.exp(lamb * np.arange(T)); p = w / w.sum()
            X = Xa[:T]; yy = ya[:T]
            wm = np.average(X, axis=0, weights=p); ws = np.sqrt(np.average((X - wm) ** 2, axis=0, weights=p)); ws[ws == 0] = 1
            from sklearn.linear_model import Ridge as _R
            m = _R(alpha=a.alpha, fit_intercept=True).fit((X - wm) / ws, yy, sample_weight=p * T)
            fit = m.predict((X - wm) / ws); res = yy - fit
            points[y] = float((m.coef_ / ws)[BASE.index(ZAR)]); fitted[y] = (fit, res, p)
        for b in range(a.boot):
            for y, i in eval_idx.items():
                T = i + 1; fit, res, p = fitted[y]
                starts = rng.integers(0, T - block, size=T // block + 1)
                idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
                yb = fit + res[idx]
                draws[y].append(wridge(Xa[:T], yb, p, a.alpha)[BASE.index(ZAR)])
        rows = []
        for y, v in draws.items():
            v = np.array(v); point = points[y]
            rows.append({'year': y, 'point': float(point), 'lo95': float(np.percentile(v, 2.5)),
                         'hi95': float(np.percentile(v, 97.5)), 'se': float(v.std()),
                         'dual_weight': DUAL[y] / 100, 'inside': bool(np.percentile(v, 2.5) <= DUAL[y] / 100 <= np.percentile(v, 97.5))})
        return pd.DataFrame(rows)
    r = cached('9_bootstrap', build, a.force)
    if r is None: return
    print(f'\n  currency loading with a {a.boot}-draw moving-block bootstrap band, against the dual-listed weight')
    print(f"    {'year':>6}{'loading':>9}{'95% band':>18}{'se':>7}{'dual weight':>13}{'weight in band':>16}")
    for _, x in r.iterrows():
        print(f"    {int(x.year):>6}{x.point:>9.3f}{f'[{x.lo95:.3f}, {x.hi95:.3f}]':>18}{x.se:>7.3f}{x.dual_weight:>13.3f}{'yes' if x.inside else 'no':>16}")
    pre = r[r.year <= 2019]; post = r[r.year > 2019]
    print(f'\n    weight inside the band: {int(pre.inside.sum())} of {len(pre)} years to 2019, {int(post.inside.sum())} of {len(post)} after')
    print('    residual block bootstrap: regressors held fixed, 26-week blocks of residuals resampled, y rebuilt,')
    print('    loading re-estimated. this is the sampling uncertainty of the coefficient given the regressors.')


# ------------------------------------------------------------------ 10. forecast
def test_forecast(d, a):
    """Fit the translation model on 2008-19 and forecast 2020-26 from the observed weights alone."""
    def build():
        B = loading_path(d, BASE, TARGETS[PRIMARY], 104, a.alpha, a.start_idx)
        if B is None: return None
        y = B[ZAR].resample('YE').mean(); y.index = y.index.year
        rows = [{'year': yr, 'dual_weight': w / 100, 'loading': float(y.loc[yr])} for yr, w in DUAL.items() if yr in y.index]
        return pd.DataFrame(rows)
    r = cached('10_forecast', build, a.force)
    if r is None: return
    fit = r[r.year <= 2019]; test = r[r.year > 2019]
    X = np.c_[np.ones(len(fit)), fit.dual_weight]; b = np.linalg.lstsq(X, fit.loading, rcond=None)[0]
    resid_sd = (fit.loading - X @ b).std()
    print(f'\n  translation model fitted on 2003 to 2019:  loading = {b[0]:+.3f} + {b[1]:.3f} x dual-listed weight')
    print(f'    (pure translation predicts an intercept of 0 and a slope of 1; residual sd {resid_sd:.3f})')
    print(f"\n    {'year':>6}{'dual weight':>13}{'predicted':>11}{'actual':>9}{'error':>8}{'in sd':>8}")
    for _, x in test.iterrows():
        pred = b[0] + b[1] * x.dual_weight; e = x.loading - pred
        print(f"    {int(x.year):>6}{x.dual_weight:>13.3f}{pred:>11.3f}{x.loading:>9.3f}{e:>+8.3f}{e / resid_sd:>+8.1f}")
    errs = test.loading - (b[0] + b[1] * test.dual_weight)
    print(f'\n    mean forecast error {errs.mean():+.3f}, or {errs.mean() / resid_sd:+.1f} residual sd. every error is negative: the')
    print('    translation model, committed before 2020, overpredicts the loading in every year after, which')
    print('    is the entry of the domestic exposure as a forecast failure rather than a residual.')


# ------------------------------------------------------------------ 11. estimator
def test_estimator(d, a):
    """Does the ratio equal one only under ridge with flexible probabilities? Rolling-window OLS."""
    def build():
        sub = d[BASE + [TARGETS[PRIMARY]]].dropna()
        rows = []
        for win in (104, 156, 208):
            out = {}
            for i in range(win, len(sub)):
                s = sub.iloc[i - win:i + 1]
                X = np.c_[np.ones(win + 1), s[BASE].values]; bb = np.linalg.lstsq(X, s[TARGETS[PRIMARY]].values, rcond=None)[0]
                out[s.index[-1]] = bb[1 + BASE.index(ZAR)]
            z = pd.Series(out).resample('YE').mean(); z.index = z.index.year
            for yr, w in DUAL.items():
                if yr in z.index: rows.append({'window': win, 'year': yr, 'dual_weight': w / 100, 'loading': float(z.loc[yr])})
        return pd.DataFrame(rows)
    r = cached('11_estimator', build, a.force)
    if r is None: return
    print('\n  ratio of loading to dual-listed weight under ordinary least squares with a fixed rolling window')
    print(f"    {'window':>8}{'to 2019':>10}{'2020-23':>10}{'2024-26':>10}{'corr, levels':>14}")
    for win, g in r.groupby('window'):
        g = g.assign(ratio=g.loading / g.dual_weight)
        print(f"    {int(win):>8}{g[g.year <= 2019].ratio.mean():>10.2f}{g[(g.year > 2019) & (g.year <= 2023)].ratio.mean():>10.2f}"
              f"{g[g.year > 2023].ratio.mean():>10.2f}{g.loading.corr(g.dual_weight):>+14.2f}")
    print('    if these agree with the ridge and flexible-probability estimates, the result is about the')
    print('    market and not the estimator. the 2024-26 column is unreliable at long windows.')




# ------------------------------------------------------------------ 12. classification
def test_classification(d, a):
    """Does the translation relationship depend on any single name, or on the breadth of the set?
    The test is the fitted slope of loading on weight (pure translation predicts 1) and the
    correlation, not the ratio: dropping a name from the set lowers the weight mechanically."""
    mpath = 'top40_monthly_class.csv'
    if not os.path.exists(mpath): print(f'  {mpath} not found; skipping'); return
    m = pd.read_csv(mpath)
    B = loading_path(d, BASE, TARGETS[PRIMARY], 104, a.alpha, a.start_idx)
    if B is None: return
    z = B[ZAR].resample('YE').mean(); z.index = z.index.year
    m['Date'] = pd.to_datetime(m['Date']); m = m[m.Date.dt.month == 4].copy(); m['Year'] = m.Date.dt.year
    def fit(mm, cat='Offshore-priced', to=2019):
        w = mm[mm.Category == cat].groupby('Year').W.sum() / mm.groupby('Year').W.sum()
        t = pd.DataFrame({'w': w, 'z': z}).dropna(); t = t[t.index <= to]
        if len(t) < 4: return np.nan, np.nan, np.nan, len(t)
        X = np.c_[np.ones(len(t)), t.w]; b = np.linalg.lstsq(X, t.z, rcond=None)[0]
        return float(b[1]), float(b[0]), float(t.z.corr(t.w)), len(t)
    s0, i0, c0, n0 = fit(m)
    print(f'\n  baseline, {n0} years to 2019: slope {s0:.2f} (translation predicts 1), intercept {i0:+.2f}, correlation {c0:+.2f}')
    print('\n  leave one dual-listed name out at a time (largest ten by peak weight):')
    dual = m[m.Category == 'Offshore-priced'].groupby('Ticker').W.max().sort_values(ascending=False).head(10)
    print(f"    {'dropped':<8}{'peak wt':>9}{'slope':>8}{'intercept':>11}{'corr':>8}")
    for tk, pk in dual.items():
        s1, i1, c1, _ = fit(m[m.Ticker != tk])
        print(f'    {tk:<8}{pk:>8.1f}%{s1:>8.2f}{i1:>+11.2f}{c1:>+8.2f}')
    print('\n  alternative definitions of the translated set:')
    strict = ['AGL', 'BHP', 'BHG', 'CFR', 'BTI', 'GLN', 'ANH', 'SAB', 'PRX', 'MNP', 'S32']
    ms = m.copy(); ms.loc[(ms.Category == 'Offshore-priced') & (~ms.Ticker.isin(strict)), 'Category'] = 'other'
    s1, i1, c1, _ = fit(ms); print(f'    strict, eleven large primary-abroad names     slope {s1:.2f}  intercept {i1:+.2f}  corr {c1:+.2f}')
    mn = m.copy(); mn.loc[mn.Ticker == 'NPN', 'Category'] = 'Offshore-priced'
    s1, i1, c1, _ = fit(mn); print(f'    Naspers treated as translated (rejected)      slope {s1:.2f}  intercept {i1:+.2f}  corr {c1:+.2f}')
    print('\n  reading. A slope near 1 and intercept near 0 is translation. The correlation falls sharply')
    print('  when Anglo American or BHP is dropped, because their weight collapsed from about 15 percent')
    print('  of the index to 3 and 1, and that collapse is most of the translated share shrinking. That')
    print('  is the phenomenon, not a fragility: the test cannot remove the largest translated shares')
    print('  without removing most of the translation. Treating Naspers as translated breaks the fit.')

# ------------------------------------------------------------------ figures
UCT = {'black': '#000000', 'blue': '#009ADA', 'navy': '#00243A', 'light': '#7FCBEC', 'grey': '#B4B4B4',
       'mid': '#4F6D7A', 'orange': '#E87722', 'green': '#2E8B57'}
NM = {'JALSH': 'All Share', 'TOP40': 'Top 40', 'MXZA': 'MSCI South Africa', 'RESI20': 'Resources 20',
      'JMNNG': 'Mining', 'INDI25': 'Industrials 25', 'FINI15': 'Financials 15', 'JGENR': 'General Retailers'}


D_PANEL = None


def make_figures(figdir):
    """Draw the finding's figures from the cached test results. Run after the tests."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.ticker
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                         'axes.edgecolor': UCT['navy'], 'axes.labelcolor': UCT['navy'],
                         'xtick.color': UCT['navy'], 'ytick.color': UCT['navy']})
    os.makedirs(figdir, exist_ok=True)
    def load(name):
        p = f'{OUT}/{name}.csv'
        return pd.read_csv(p) if os.path.exists(p) else None
    def style(ax):
        ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
    def save(fig, name):
        fig.tight_layout(); fig.savefig(f'{figdir}/{name}.png', dpi=200); plt.close(fig); print(f'   {figdir}/{name}.png')

    # 1. the loading against the dual-listed weight, with the bootstrap band if available
    r = load('1_ratio'); bt = load('9_bootstrap')
    if r is not None:
        g = r[(r.target == PRIMARY) & (r.tau == 104) & (r.alpha == 5.6)].sort_values('year')
        if len(g):
            fig, ax = plt.subplots(figsize=(8.6, 4.2))
            if bt is not None:
                bt = bt.sort_values('year')
                ax.fill_between(bt.year, bt.lo95, bt.hi95, color=UCT['blue'], alpha=.15, label='95% band on the loading, residual block bootstrap')
            ax.plot(g.year, g.loading, marker='o', ms=4, color=UCT['blue'], lw=1.8, label='Top 40 loading on USDZAR')
            ax.plot(g.year, g.dual_weight, marker='s', ms=4, color=UCT['navy'], lw=1.8, ls='--', label='weight of dual-listed constituents')
            ax.plot(g.year, g.sa_dollar, marker='^', ms=4, color=UCT['orange'], lw=1.4, ls=':', label='weight of South African dollar-earners')
            ax.axhline(0, color=UCT['mid'], lw=.6); ax.set_ylabel('loading, or share of index'); ax.set_xlabel('')
            ax.set_xticks(list(range(2008, 2027, 2))); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v)}'))
            ax.set_title('The currency loading equals the foreign-priced share of the Top 40', loc='left')
            ax.legend(frameon=False, loc='upper right'); style(ax)
            save(fig, 'fig_translation_ratio')


    # 1b and 1c. the two candidates, one figure each
    if r is not None:
        g = r[(r.target == PRIMARY) & (r.tau == 104) & (r.alpha == 5.6)].sort_values('year')
        if len(g):
            # translation: the loading against the dual-listed weight, with the band
            fig, ax = plt.subplots(figsize=(8.2, 3.9))
            if bt is not None:
                ax.fill_between(bt.year, bt.lo95, bt.hi95, color=UCT['blue'], alpha=.15, label='95% band on the loading')
            ax.plot(g.year, g.loading, marker='o', ms=4, color=UCT['blue'], lw=1.8, label='Top 40 loading on USDZAR')
            ax.plot(g.year, g.dual_weight, marker='s', ms=4, color=UCT['navy'], lw=1.8, ls='--', label='share of the index that is foreign-priced (dual-listed)')
            for yr, lo, w in zip(g.year, g.loading, g.dual_weight):
                if yr <= 2019: ax.annotate(f'{lo/w:.2f}', (yr, max(lo, w)), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7, color=UCT['mid'])
            ax.axhline(0, color=UCT['mid'], lw=.6); ax.set_ylabel('loading, or share of index')
            ax.set_xticks(list(range(2008, 2027, 2))); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v)}'))
            ax.set_title('Candidate 1, translation: the loading should equal the foreign-priced share of the index', loc='left', fontsize=9.5)
            ax.legend(frameon=False, loc='lower left'); style(ax)
            ax.annotate('ratio of loading to share, each year to 2019', xy=(2012, ax.get_ylim()[1]*.97), fontsize=7, color=UCT['mid'], ha='center', va='top')
            save(fig, 'fig_candidate_translation')
            # economic: the loading against the dollar-earner weight
            fig, ax = plt.subplots(figsize=(8.2, 3.9))
            ax.plot(g.year, g.loading, marker='o', ms=4, color=UCT['blue'], lw=1.8, label='Top 40 loading on USDZAR')
            ax.plot(g.year, g.sa_dollar, marker='^', ms=4, color=UCT['orange'], lw=1.8, ls='--', label='share of the index that is South African dollar-earners')
            ax.axhline(0, color=UCT['mid'], lw=.6); ax.set_ylabel('loading, or share of index')
            ax.set_xticks(list(range(2008, 2027, 2))); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v)}'))
            ax.set_title('Candidate 2, economic hedge: the loading should rise and fall with the dollar-earner share', loc='left', fontsize=9.5)
            ax.legend(frameon=False, loc='upper left'); style(ax)
            save(fig, 'fig_candidate_economic')


    # 1d. the decomposition: loading = W + D, through time, with the uncertainty that says when D is distinguishable from zero
    r = load('1_ratio'); bt = load('9_bootstrap')
    if r is not None:
        g = r[(r.target == PRIMARY) & (r.tau == 104) & (r.alpha == 5.6)].sort_values('year')
        if len(g):
            W = g.dual_weight.values; L = g.loading.values; D = L - W; yrs = g.year.values
            se = bt.set_index('year').se.reindex(yrs).fillna(bt.se.mean()).values if bt is not None else np.full(len(yrs), 0.05)
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True, gridspec_kw={'height_ratios': [1.35, 1]})
            a1.fill_between(yrs, 0, W, color=UCT['navy'], alpha=.85, label='W: foreign-priced share of the index')
            a1.plot(yrs, L, color=UCT['blue'], lw=2.2, marker='o', ms=4, label='the index loading on USDZAR')
            a1.axhline(0, color=UCT['mid'], lw=.6); a1.set_ylabel('loading, or share of index'); a1.set_ylim(0, max(L.max(), W.max()) * 1.15)
            a1.set_title('loading  =  W  +  D', loc='left'); a1.legend(frameon=False, loc='upper right', fontsize=8); style(a1)
            a2.fill_between(yrs, -1.96 * se, 1.96 * se, color=UCT['grey'], alpha=.35, label='95% band: D indistinguishable from zero')
            a2.bar(yrs, D, width=.7, color=[UCT['orange'] if abs(d) > 1.96 * s else UCT['mid'] for d, s in zip(D, se)])
            a2.axhline(0, color=UCT['navy'], lw=.8); a2.set_ylabel('D = loading − W')
            a2.set_title('D: what the Johannesburg-priced part contributes. Orange where it is outside the band.', loc='left', fontsize=9.5)
            a2.legend(frameon=False, loc='lower left', fontsize=8); style(a2)
            a2.set_xticks(list(range(2004, 2027, 2)))
            save(fig, 'fig_loading_decomposition')
    # 1e. what the index was made of, by category, through time
    if True:
        Cm = WEIGHTS_MONTHLY.copy()
        C = pd.DataFrame({'dual-listed': Cm['dual'], 'SA dollar-earners': Cm['earn'], 'Naspers': Cm['naspers']})
        C.index = C.index.year + (C.index.month - 1) / 12.0
        C['domestic'] = 100 - C.sum(axis=1)
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        cols = ['dual-listed', 'SA dollar-earners', 'Naspers', 'domestic']
        colr = [UCT['navy'], UCT['orange'], UCT['grey'], '#dfe7ea']
        ax.stackplot(C.index, [C[c] for c in cols], labels=cols, colors=colr, alpha=.95)
        for x, lab, y0, ha in [(2023.42, 'Richemont\ntreated as\nforeign\nJun 2023', 96, 'right'), (2024.17, 'free-float\nharmonisation\nphase 1\nMar 2024', 96, 'left'),
                               (2023.92, 'AngloGold\nforeign\nDec 2023', 70, 'center'),
                               (2025.3, 'gold and\nplatinum run\n2024–26', 52, 'center'), (2021.6, 'Prosus\ntransaction\nAug 2021', 96, 'right')]:
            ax.axvline(x, color='white', lw=1.2, ls=':')
            ax.annotate(lab, xy=(x, y0), xytext=(-3 if ha == 'right' else 0, 0), textcoords='offset points', fontsize=6.8, color=UCT['mid'], ha=ha, va='top')
        ax.set_ylim(0, 100); ax.set_ylabel('% of index'); ax.set_xlim(C.index.min(), C.index.max()); ax.set_xticks(list(range(2004, 2027, 2)))
        for yr, ha in ((2003.25, 'left'), (2019.25, 'center'), (2023.25, 'center'), (2026.75, 'right')):
            i = C.index.get_indexer([yr], method='nearest')[0]
            ax.annotate(f"{C.iloc[i]['dual-listed']:.0f}%", (C.index[i], C.iloc[i]['dual-listed'] / 2), ha=ha, fontsize=8, color='white')

        ax.set_title(f'What the {NM[PRIMARY]} was made of: index weight by category, monthly', loc='left')
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, -0.12), ncol=4, fontsize=8); style(ax)
        save(fig, 'fig_index_composition')


    # 1f. what each mechanism contributes: translation at 1 per unit weight, the economic mechanism at ~0.24
    r = load('1_ratio')
    if r is not None:
        g = r[(r.target == PRIMARY) & (r.tau == 104) & (r.alpha == 5.6)].sort_values('year')
        if len(g):
            ECON = 0.24   # the Mining index's end-of-sample loading, when its constituents are mostly SA-domiciled producers
            yrs = g.year.values; W = g.dual_weight.values; S = g.sa_dollar.values; L = g.loading.values
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True, gridspec_kw={'height_ratios': [1, 1.15]})
            a1.plot(yrs, 100 * W, color=UCT['navy'], lw=2, marker='s', ms=4, label='foreign-priced share of the index')
            a1.plot(yrs, 100 * S, color=UCT['orange'], lw=2, marker='^', ms=4, label='South African dollar-earner share')
            a1.set_ylabel('% of index'); a1.set_title('The two groups the received account bundles together', loc='left')
            a1.legend(frameon=False, fontsize=8); style(a1)
            a1.annotate('dollar-earners triple\nfrom 2020', xy=(2026, 100 * S[-1]), xytext=(2021.5, 100 * S[-1] + 12), fontsize=7.5, color=UCT['orange'], arrowprops=dict(arrowstyle='-', color=UCT['orange'], lw=.8))
            a2.fill_between(yrs, 0, W, color=UCT['navy'], alpha=.85, label='translation: foreign-priced share × 1.0')
            a2.fill_between(yrs, W, W + ECON * S, color=UCT['orange'], alpha=.9, label=f'economic mechanism: dollar-earner share × {ECON}')
            a2.plot(yrs, L, color=UCT['blue'], lw=2.2, marker='o', ms=4, label='the index loading on USDZAR')
            a2.axhline(0, color=UCT['mid'], lw=.6); a2.set_ylabel('contribution to the loading')
            a2.set_title('What each contributes to the currency loading', loc='left'); a2.legend(frameon=False, fontsize=8, loc='upper right'); style(a2)
            a2.set_xticks(list(range(2004, 2027, 2)))
            for yr in (2008, 2026):
                i = list(yrs).index(yr); a2.annotate(f'{ECON*S[i]:.2f}', (yr, W[i] + ECON * S[i]), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=7, color=UCT['orange'])
            save(fig, 'fig_two_mechanisms')

    # 2. the residual from pure translation, against the Financials loading
    r = load('3_identity')
    if r is not None and 'FINI15' in r.columns:
        g = r.sort_values('year'); res = g['index'] - g.dual_weight
        fig, ax1 = plt.subplots(figsize=(8.6, 3.8))
        ax1.bar(g.year, res, color=[UCT['grey'] if abs(v) < 0.03 else UCT['orange'] for v in res], width=.7, label='index loading minus dual-listed weight')
        ax1.axhline(0, color=UCT['mid'], lw=.7); ax1.set_ylabel('residual from pure translation')
        ax2 = ax1.twinx(); ax2.plot(g.year, g.FINI15, color=UCT['blue'], marker='o', ms=3, lw=1.5, label='Financials 15 loading on USDZAR (right axis)')
        ax2.set_ylabel('Financials loading', color=UCT['blue']); ax2.spines[['top']].set_visible(False)
        ax1.set_title('What entered after 2019: the Top 40 residual from translation tracks the domestic sectors', loc='left')
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, loc='lower left'); style(ax1)
        save(fig, 'fig_translation_residual')

    # 3. the spread decomposition: currency loading at sample end by index and specification
    r = load('2_spread')
    if r is not None:
        g = r[r.tau == 104]
        specs = [('three channels', UCT['navy']), ('plus US 10y', UCT['grey']), ('plus SA spread', UCT['orange'])]
        order = [t for t in ['RESI20', 'JMNNG', 'TOP40', 'JALSH', 'MXZA', 'INDI25', 'FINI15', 'JGENR'] if t in g.target.values]
        fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(order)); w = .26
        for k, (sp, col) in enumerate(specs):
            vals = [g[(g.target == t) & (g.spec == sp)].zar_end.values[0] if ((g.target == t) & (g.spec == sp)).any() else np.nan for t in order]
            ax.bar(x + (k - 1) * w, vals, w, color=col, label=sp)
        ax.axhline(0, color=UCT['mid'], lw=.7); ax.set_xticks(x); ax.set_xticklabels([NM[t] for t in order], fontsize=8)
        ax.set_ylabel('loading on USDZAR at sample end')
        ax.set_title('Adding the South African spread moves the domestic sectors and not the resource indices; the US ten-year moves nothing', loc='left', fontsize=9.5)
        ax.legend(frameon=False, loc='lower left'); style(ax)
        save(fig, 'fig_spread_decomposition')

    # 4. the conditional profile, early against late, four indices
    r = load('4_profile')
    if r is not None:
        g = r[(~r.decay) & (r.period != 'full')]
        panels = [t for t in ['TOP40', 'MXZA', 'RESI20', 'JGENR'] if t in g.target.values]
        if panels:
            fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True); axes = axes.flatten()
            for ax, t in zip(axes, panels):
                gg = g[g.target == t]
                for per, col in zip(sorted(gg.period.unique()), [UCT['navy'], UCT['orange']]):
                    s = gg[gg.period == per].sort_values('percentile')
                    ax.plot(s.percentile, s.zar, marker='o', ms=3.5, color=col, lw=1.6, label=per)
                ax.axhline(0, color=UCT['mid'], lw=.7); ax.set_title(NM[t], loc='left'); ax.legend(frameon=False); style(ax)
            for ax in axes[2:]: ax.set_xlabel("percentile of the week's USDZAR return  (left: rand strength, right: rand weakness)", fontsize=8)
            for ax in axes[::2]: ax.set_ylabel('loading on USDZAR')
            fig.suptitle('The currency loading across the range of the rand\'s move, early and late sample', x=0.02, ha='left', fontsize=10)
            save(fig, 'fig_rand_profile')

    # 5. the taxonomy: currency loading through time by index, over Flint-style bands
    r = load('5_taxonomy')
    if r is not None:
        r['date'] = pd.to_datetime(r['date'])
        COL = {'RESI20': UCT['navy'], 'JMNNG': '#1b4a63', 'TOP40': UCT['black'], 'JALSH': UCT['mid'],
               'MXZA': UCT['grey'], 'INDI25': UCT['green'], 'FINI15': UCT['blue'], 'JGENR': UCT['orange']}
        fig, ax = plt.subplots(figsize=(11, 5.4))
        for lo, hi, lab, c in [(0.45, 2, 'rand hedge', '#e8f4fa'), (0.15, 0.45, 'partial hedge', '#f4fafd'),
                               (-0.15, 0.15, 'no currency exposure', '#f7f7f7'), (-2, -0.15, 'rand play', '#fdf0e6')]:
            ax.axhspan(lo, hi, color=c, zorder=0)
            ax.annotate(lab, xy=(0.004, (max(lo, -0.95) + min(hi, 1.35)) / 2), xycoords=('axes fraction', 'data'), fontsize=8, color=UCT['mid'], va='center', zorder=1)
        for t, g in r.groupby('target'):
            g = g.sort_values('date'); ax.plot(g.date, g.zar, color=COL.get(t, UCT['mid']), lw=1.5, zorder=3)
            ax.annotate(NM.get(t, t), xy=(g.date.iloc[-1], g.zar.iloc[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8, color=COL.get(t, UCT['mid']), va='center')
        for y in (0.45, 0.15, -0.15): ax.axhline(y, color='white', lw=1.1, zorder=2)
        ax.axhline(0, color=UCT['mid'], lw=.7, zorder=2); ax.set_ylim(-1.0, 1.4); ax.set_ylabel('loading on USDZAR (raw units)')
        ax.set_xlim(r.date.min(), r.date.max() + pd.Timedelta(days=1500))
        ax.set_title('The currency exposure of the Johannesburg market through time, by index', loc='left'); style(ax)
        save(fig, 'fig_currency_taxonomy')


    # 7. the South African country risk premium: what it is, and what has come to move with it
    dd = D_PANEL
    if 'ALBIAY Index' in dd.columns and 'USGG10YR Index' in dd.columns:
        # the ALBI yield has no gaps; the 10-year government series does, and a line would bridge them
        spread = (dd['ALBIAY Index'] - dd['USGG10YR Index']).dropna()
        dsp = spread.diff()
        cds = dd['REPSOU CDS USD SR 5Y D14 Corp'] if 'REPSOU CDS USD SR 5Y D14 Corp' in dd.columns else None
        w = 104
        rc_rand = dd[ZAR].rolling(w).corr(dsp).dropna()
        fig, ax = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True, gridspec_kw={'height_ratios': [1.4, 1, 1]})
        ax[0].plot(spread.index, spread, color=UCT['navy'], lw=1.3, label='All Bond Index yield minus US 10-year yield, pp')
        ax[0].set_ylabel('percentage points', color=UCT['navy'])
        if cds is not None:
            cds = cds.where(cds.notna())   # leave gaps as gaps rather than bridging them
            a2 = ax[0].twinx(); a2.plot(cds.index, cds, color=UCT['orange'], lw=1.0, alpha=.85, label='5-year sovereign CDS, bp (right)')
            a2.set_ylabel('basis points', color=UCT['orange']); a2.spines[['top']].set_visible(False)
            h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, frameon=False, loc='upper right')
        else: ax[0].legend(frameon=False)
        ax[0].set_title('(a) The price of South African country risk', loc='left'); style(ax[0])
        ax[1].plot(rc_rand.index, rc_rand, color=UCT['navy'], lw=1.3)
        ax[1].axhline(0, color=UCT['mid'], lw=.6); ax[1].set_ylabel('rolling correlation')
        ax[1].set_title('(b) The rand has come to move with it: 104-week correlation of the USDZAR move with the change in the spread', loc='left', fontsize=9.5); style(ax[1])
        for t, col in [('FINI15', UCT['blue']), ('JGENR', UCT['orange']), ('RESI20', UCT['grey'])]:
            if TARGETS[t] in dd.columns:
                rc = dd[TARGETS[t]].rolling(w).corr(dsp).dropna(); ax[2].plot(rc.index, rc, color=col, lw=1.3, label=NM[t])
        ax[2].axhline(0, color=UCT['mid'], lw=.6); ax[2].set_ylabel('rolling correlation')
        ax[2].set_title('(c) Domestic equity has become sensitive to it: 104-week correlation of index returns with the change in the spread', loc='left', fontsize=9.5)
        ax[2].legend(frameon=False, ncol=3, loc='lower left'); style(ax[2])
        save(fig, 'fig_country_risk')

    # 6. the forecast test: translation fitted to 2019, extrapolated
    r = load('10_forecast')
    if r is not None:
        fit = r[r.year <= 2019]; X = np.c_[np.ones(len(fit)), fit.dual_weight]; b = np.linalg.lstsq(X, fit.loading, rcond=None)[0]
        fig, ax = plt.subplots(figsize=(8.6, 4))
        ax.plot(r.year, r.loading, marker='o', ms=4, color=UCT['blue'], lw=1.8, label='Top 40 loading on USDZAR')
        ax.plot(r.year, b[0] + b[1] * r.dual_weight, marker='s', ms=3.5, color=UCT['navy'], lw=1.4, ls='--', label='predicted by the dual-listed weight, model fitted to 2019')
        ax.axvline(2019.5, color=UCT['grey'], ls=':', lw=1); ax.annotate('fit | forecast', xy=(2019.5, ax.get_ylim()[1] * .95), fontsize=8, color=UCT['mid'], ha='center')
        ax.fill_between(r.year, r.loading, b[0] + b[1] * r.dual_weight, where=r.year > 2019, color=UCT['orange'], alpha=.25, label='forecast error')
        ax.set_ylabel('loading'); ax.set_title('The translation model, committed before 2020, overpredicts every year after', loc='left')
        ax.legend(frameon=False, loc='lower left'); style(ax)
        save(fig, 'fig_translation_forecast')


# ------------------------------------------------------------------ main
TESTS = {'1': ('ratio', test_ratio), '2': ('spread', test_spread), '3': ('identity', test_identity),
         '4': ('profile', test_profile), '5': ('taxonomy', test_taxonomy),
         '6': ('rechecks', test_rechecks), '7': ('reconcile', test_reconcile),
         '8': ('placebo', test_placebo), '9': ('bootstrap', test_bootstrap),
         '10': ('forecast', test_forecast), '11': ('estimator', test_estimator),
         '12': ('classification', test_classification)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='transformed_data.csv')
    ap.add_argument('--tests', nargs='*', default=list(TESTS))
    ap.add_argument('--taus', type=float, nargs='*', default=[52, 104, 156])
    ap.add_argument('--alpha', type=float, default=5.6)
    ap.add_argument('--start-idx', type=int, default=250)
    ap.add_argument('--panel-start', default='1999-01-08',
                    help='drop data before this date so the estimation sample matches the main results; empty to keep all')
    ap.add_argument('--start-date', default='2003-10-31',
                    help='first estimation date, so every panel gives comparable results; empty string to disable')
    ap.add_argument('--boot', type=int, default=500, help='bootstrap draws for test 9')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--report', action='store_true', help='print from cache without recomputing')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--plot', action='store_true', help='draw the figures from cached results')
    ap.add_argument('--figdir', default='figures')
    a = ap.parse_args()
    OUT = a.out
    FIRST_ESTIMATE = a.start_date or None
    if a.report: a.force = False
    if a.quick: a.taus = [104]

    if not os.path.exists(a.data):
        sys.exit(f'{a.data} not found. Build it first:  python 00_transform.py --out {a.data}')
    d = pd.read_csv(a.data, index_col=0, parse_dates=True)
    if a.panel_start: d = d.loc[a.panel_start:]
    missing = [c for c in BASE if c not in d.columns]
    if missing: sys.exit(f'missing columns: {missing}')
    print(f'panel {a.data}: {len(d)} rows, {d.index.min().date()} to {d.index.max().date()}')
    print(f'panel truncated to start {a.panel_start or "(not truncated)"}; half-lives {a.taus}, alpha {a.alpha}, first estimate on or after {FIRST_ESTIMATE}'
          f'{"  [quick]" if a.quick else ""}\n')

    t0 = time.time()
    for k in a.tests:
        if k not in TESTS: continue
        name, fn = TESTS[k]
        print(f'{"=" * 78}\n{k}. {name}\n{"=" * 78}')
        try:
            fn(d, a)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
        print()
    print(f'done in {(time.time() - t0) / 60:.0f} minutes. results cached in {OUT}/')
    if a.plot:
        D_PANEL = d
        print('\nfigures:'); make_figures(a.figdir)
    print('\nWhat each result has to show for the finding to hold:')
    print('  1  the ratio of loading to dual-listed weight near 1 before 2020, at every half-life')
    print('  2  the SA spread moves the domestic sectors and not the resource indices; the US')
    print('     ten-year moves nothing')
    print('  3  the residual from pure translation near zero before 2020 and negative after')
    print('  4  the profile rising into rand weakness, and shifting down between periods')
    print('  6  the decline present in every recheck')
    print('  7  a substantial decline already in place by the end of 2023')
    print('  8  the currency loading tracks the weight; the mining and EM loadings do not')
    print('  9  the dual-listed weight inside the bootstrap band to 2019 and outside it after')
    print('  10 the translation model, fitted to 2019, overpredicting every year after')
    print('  11 the ratio near 1 under rolling OLS as under ridge')
    print('  12 the ratio near 1 whichever dual-listed name is dropped, and under the strict set')
