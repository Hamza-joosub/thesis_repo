"""01_validate.py — harness. Run before anything else on a freshly frozen data file.
Checks the estimator against textbook oracles and the FP weights against an independent implementation.
    python 01_validate.py [data/thesis_data.csv]
"""
import sys, hashlib, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings('ignore')
sys.argv.append(None) if len(sys.argv) < 2 else None
CSV = sys.argv[1] or 'data/thesis_data.csv'
import importlib.util
spec = importlib.util.spec_from_file_location('runner', '02_run_specs.py'); R = importlib.util.module_from_spec(spec)
sys.argv = ['x']; spec.loader.exec_module(R)                  # loads the model functions without running main
fit = R.fit_ridge_FP_SC
df = pd.read_csv(CSV); df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date')
print(f"file {CSV}: {df.shape}, {df.index.min().date()} -> {df.index.max().date()}, NaN={int(df.isna().sum().sum())}")
print("sha256", hashlib.sha256(open(CSV,'rb').read()).hexdigest())
X = ['MXWO0MM Index_Log_Diff','MXEF Index_Log_Diff','MXWO0CS Index_Log_Diff','USDZAR Curncy_Log_Diff']; y = 'JALSH Index_Log_Diff'
d = df.iloc[:600].copy(); T = len(d); lamb = np.log(2)/104; ok = True
def chk(name, val, tol=1e-9):
    global ok; p = val < tol; ok &= p; print(f"  {'PASS' if p else 'FAIL'}  {name}: {val:.2e}")
r = fit(d, lamb=None, alpha=1e-10, X_cols=list(X), y_col=y, conditioning_variable=None, conditioning_percentile=None)
ols = sm.OLS(d[y], sm.add_constant(d[X])).fit(); chk("OLS limit (uniform weights, alpha->0)", max(abs(r['coefficients']['raw']['betas'][c]-ols.params[c]) for c in X))
r = fit(d, lamb=lamb, alpha=1e-10, X_cols=list(X), y_col=y, conditioning_variable=None, conditioning_percentile=None)
w = np.exp(lamb*np.arange(T)); w /= w.sum(); wls = sm.WLS(d[y], sm.add_constant(d[X]), weights=w).fit()
chk("WLS limit (exponential weights, alpha->0)", max(abs(r['coefficients']['raw']['betas'][c]-wls.params[c]) for c in X))
chk("ENS = exp(-sum p log p)", abs(r['Stage1_diagnostics']['ens'] - np.exp(-np.sum(w*np.log(w)))))
# brute-force LOO under the closed form's assumptions
from sklearn.linear_model import Ridge
r = fit(d, lamb=lamb, alpha=5.6, X_cols=list(X), y_col=y, conditioning_variable=None, conditioning_percentile=None)
Xr = d[X].values; yr = d[y].values; mu = np.average(Xr, axis=0, weights=w); sd = np.sqrt(np.average((Xr-mu)**2, axis=0, weights=w)); Xs = (Xr-mu)/sd
pred = np.empty(T)
for i in range(T):
    m = np.ones(T, bool); m[i] = False
    pred[i] = Ridge(alpha=5.6).fit(Xs[m], yr[m], sample_weight=(w*T)[m]).predict(Xs[[i]])[0]
ybar = np.average(yr, weights=w); loo_bf = 1 - np.sum(w*(yr-pred)**2)/np.sum(w*(yr-ybar)**2)
chk("closed-form LOO = brute-force LOO", abs(r['fit']['LOO_r2'] - loo_bf))
# lookahead
date = d.index[-1]; d2 = df.copy(); d2.loc[d2.index > date, X+[y]] = 999.0
ra = fit(df.loc[:date].copy(), lamb=lamb, alpha=5.6, X_cols=list(X), y_col=y, conditioning_variable=None, conditioning_percentile=None)
rb = fit(d2.loc[:date].copy(), lamb=lamb, alpha=5.6, X_cols=list(X), y_col=y, conditioning_variable=None, conditioning_percentile=None)
chk("no lookahead (future rows corrupted, fit unchanged)", max(abs(ra['coefficients']['raw']['betas'][c]-rb['coefficients']['raw']['betas'][c]) for c in X), 1e-15)
# Matthews (2024) es_probs / ens, transcribed from her MATLAB (verified verbatim under Octave, see VALIDATION.md)
def es_probs(Tn, tau): t = np.arange(1, Tn+1); p = np.exp(-np.log(2)/tau*(Tn-t)); return p/p.sum()
def ens_m(p): p = np.maximum(p, 1e-250); return np.exp(-np.sum(p*np.log(p)))
chk("ENS vs Matthews es_probs+ens at T=1060, tau=118", abs(ens_m(es_probs(1060, 118)) - 456.1818197701), 1e-6)
# stationarity
from statsmodels.tsa.stattools import adfuller
tf = [c for c in df.columns if c.endswith(('_Log_Diff','_Diff'))]; ps = [adfuller(df[c], autolag='AIC')[1] for c in tf]
print(f"  ADF: {sum(p<0.05 for p in ps)}/{len(ps)} transformed series stationary at 5% (max p = {max(ps):.1e})")
print("\nALL PASS" if ok else "\nFAILURES — do not run the specs until resolved")
