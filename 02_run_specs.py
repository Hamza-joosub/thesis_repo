"""
run_specs.py — run many ridge-FP specifications against many targets and store everything.

Two tiers:
  screen : EVERY subset (size 2..MAX_K) of FACTOR_POOL, evaluated on a coarse date grid
           (every STEP-th estimation date) on the primary target. Cheap. Ranks by mean LOO R².
  full   : full-resolution rolling fits (every estimation date) for CURATED specs plus the
           top TOP_K screen specs, on EVERY target. Stores per run:
             betas_raw.csv, betas_std.csv        loading paths (raw units / standardised)
             fit.csv                             r2_weighted, r2_unweighted, LOO_r2, alpha, ENS per date
             contrib_weekly.csv                  signed weekly contribution  beta_raw[t] * x[t]  (NOT aggregated)
           plus results/summary.csv and results/spec_registry.json.

Usage:
  python run_specs.py --screen                 # exhaustive screen on JALSH
  python run_specs.py --full                   # full runs, curated + top-K, all targets
  python run_specs.py --screen --full          # both
  python run_specs.py --full --targets JALSH RESI20 --specs baseline monolithic
Options: --tau 118  --alpha auto|<float>  --step 26  --max-k 5  --top-k 8  --start-idx 250
"""

# ============================ MODEL (thesis_model.py, app imports removed) ============================
"""
Thesis sandbox — ridge regression with exponential FP weighting.

Research tool, not a production model: no caching, no scheduler, runs once
per HTTP call. fit_ridge_FP and rolling_fit_ridge_FP are copied from the
notebook verbatim except the dependent variable name is now a parameter
(y_col) so target/predictors can be chosen at request time.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import json
import os
from scipy.optimize import brentq
from scipy.stats import jarque_bera

from scipy.optimize import brentq



CSV_PATH = None   # set by the runner from CSV / --csv at startup; never edit here
DATA_CLEANING_REPORT_PATH = 'data_cleaning_report.json'

def get_data_cleaning_report():
    with open(DATA_CLEANING_REPORT_PATH, 'r') as f:
        return json.load(f)

def get_raw_columns():
    df = pd.read_csv(CSV_PATH)
    return [col for col in df.columns if not col.endswith('_Diff') and not col.endswith('_Log_Diff')]

def get_transformed_columns():
    df = pd.read_csv(CSV_PATH)
    return [col for col in df.columns if col.endswith('_Diff') or col.endswith('_Log_Diff')]
    
    
def state_conditional_fp(p_prior, conditioning_values, target_value):
    scale = max(abs(conditioning_values.max()), abs(conditioning_values.min()), 1)
    bound = 5.0 / scale
    
    def constraint_error(lam):
        log_tilt = lam * conditioning_values
        log_tilt -= log_tilt.max()  # numerical stability
        p_tilt = p_prior * np.exp(log_tilt)
        s = p_tilt.sum()
        if s == 0 or np.isnan(s):
            return np.nan
        p_tilt /= s
        return np.sum(p_tilt * conditioning_values) - target_value
    
    try:
        lam_star = brentq(constraint_error, -bound, bound)
    except Exception:
        return p_prior, np.exp(-np.sum(p_prior * np.log(p_prior))), 0.0
    
    log_tilt = lam_star * conditioning_values
    log_tilt -= log_tilt.max()
    p_posterior = p_prior * np.exp(log_tilt)
    p_posterior /= p_posterior.sum()
    ens = np.exp(-np.sum(p_posterior * np.log(p_posterior)))
    
    return p_posterior, ens, lam_star

def fit_ridge_FP_SC(df, lamb, alpha, X_cols,y_col, conditioning_variable, conditioning_percentile ):
    if conditioning_variable is not None and conditioning_variable not in df.columns:
        raise ValueError(f"conditioning variable: {conditioning_variable} not in dataframe")

    df_Ridge_TimeFP = df.reset_index().reset_index().set_index('Date').rename(columns={'index': 't'})
    T = len(df_Ridge_TimeFP)
    
    # Temporal weights
    if lamb is not None:
        df_Ridge_TimeFP['Exponential Weights'] = np.exp(lamb * df_Ridge_TimeFP['t'])
        df_Ridge_TimeFP['Exponential Probability'] = (
            df_Ridge_TimeFP['Exponential Weights'] / df_Ridge_TimeFP['Exponential Weights'].sum()
        )
    else:
        df_Ridge_TimeFP['Exponential Probability'] = 1.0 / T
    
    p1 = df_Ridge_TimeFP['Exponential Probability'].values
    
    shannon_entropy = -np.sum(p1 * np.log(p1))
    ens = np.exp(shannon_entropy)
    
    # State conditioning
    if conditioning_variable is not None:
        cond_values = df_Ridge_TimeFP[conditioning_variable].values
        target = np.percentile(cond_values, conditioning_percentile)
        p, ens_stage2, lam_s = state_conditional_fp(p1, cond_values, target)
        # state_conditional_fp returns the prior unchanged when brentq fails
        # to find a feasible lambda. Short-circuit with INFEASIBLE so the
        # rolling loop drops this date rather than reporting a fit that
        # *looks* state-conditioned but is actually the time-only prior.
        if p is p1:
            return {
                "params": {
                    "lambda_temporal":         float(lamb) if lamb is not None else None,
                    "alpha":                   float(alpha),
                    "conditioning_variable":   conditioning_variable,
                    "conditioning_percentile": conditioning_percentile,
                    "conditioning_status":     "INFEASIBLE",
                    "T":                       int(T),
                }
            }
        conditioning_status = "OK"
    else:
        p = p1
        ens_stage2 = ens
        lam_s = 0.0
        conditioning_status = "OK"
    
    X = df_Ridge_TimeFP[X_cols].values
    y = df_Ridge_TimeFP[y_col].values
    p1 = df_Ridge_TimeFP['Exponential Probability'].values
    T = len(p1)


    
    weighted_mean = np.average(X, axis=0, weights=p)
    weighted_var = np.average((X - weighted_mean) ** 2, axis=0, weights=p)
    weighted_std = np.sqrt(weighted_var)
    
    simple_mean = X.mean(axis=0)
    simple_std = X.std(axis=0)
    
    X_std = (X - weighted_mean) / weighted_std
    
    standardisation_check = {}
    for j, col in enumerate(X_cols):
        wm = np.average(X_std[:, j], weights=p)
        ws = np.sqrt(np.average((X_std[:, j] - wm) ** 2, weights=p))
        standardisation_check[col] = {"mean": float(wm), "std": float(ws)}
    
    p_rescaled = p * T
    
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_std, y, sample_weight=p_rescaled)

    
    
    beta_std = model.coef_
    intercept_std = model.intercept_
    y_pred = X_std @ beta_std + intercept_std # @ is dot 
    resid = y-y_pred
    
    ## Leave One out Cross Validation
    
    X_aug = np.column_stack([np.ones(T), X_std]) # adds a bunch 1 to the 1st columns
    k_aug = X_aug.shape[1]

    penalty = alpha * np.eye(k_aug) #np.eye is an I matrix and we bascially only apply alpha on beta
    penalty[0, 0] = 0.0

    A = X_aug.T @ (p_rescaled[:, None] * X_aug) + penalty   # (k+1, k+1)
    A_inv = np.linalg.solve(A, np.eye(k_aug))

    XA = X_aug @ A_inv
    h_diag = np.sum(XA * X_aug, axis=1) * p_rescaled

    loo_resid = resid / (1 - h_diag)

    press_w = np.sum(p * loo_resid ** 2)
    ss_tot_w = np.sum(p * (y - np.average(y, weights=p)) ** 2)
    r2_loo = 1 - press_w / ss_tot_w



    beta_raw = beta_std / weighted_std
    intercept_raw = intercept_std - np.sum(beta_std * weighted_mean / weighted_std)
    
    betas_raw_dict = {col: float(b) for col, b in zip(X_cols, beta_raw)}
    betas_raw_sorted = dict(sorted(betas_raw_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    ss_res = np.sum(p * (y - y_pred) ** 2)
    ss_tot = np.sum(p * (y - np.average(y, weights=p)) ** 2)
    r2_weighted = 1 - ss_res / ss_tot
    
    ss_res_u = np.sum((y - y_pred) ** 2)
    ss_tot_u = np.sum((y - y.mean()) ** 2)
    r2_unweighted = 1 - ss_res_u / ss_tot_u

    # Residual diagnostics
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
    jb_stat, jb_p = jarque_bera(resid)


    
    ridge_report = {
            "params": {
            "lambda_temporal": float(lamb) if lamb is not None else None,  # FIX: uniform-weight (OLS baseline) runs crashed here
            "half_life_weeks": float(np.log(2)/lamb) if lamb else None,     # NEW: report tau alongside the rate — thesis speaks in half-lives
            "lambda_state": float(lam_s),
            "alpha": float(alpha),
            "conditioning_variable": conditioning_variable,
            "conditioning_percentile": conditioning_percentile,
            "conditioning_status": conditioning_status,
            "T": int(T),
        },
        "Stage1_diagnostics": {
            "ens":                     float(ens),
            "shannon_entropy":         float(shannon_entropy),
            "weight_oldest":           float(df_Ridge_TimeFP['Exponential Probability'].iloc[0]),
            "weight_sum":              float(df_Ridge_TimeFP['Exponential Probability'].sum()),
            "cumulative_weight_52w":   float(df_Ridge_TimeFP['Exponential Probability'].iloc[-52:].sum()),
            "cumulative_weight_104w":  float(df_Ridge_TimeFP['Exponential Probability'].iloc[-104:].sum()),
            "lambda":                   float(lamb) if lamb is not None else None,  # FIX: None-safe
            "weights_Stage_1":                pd.Series(p1)

        },
        "Stage2_diagnostics": {
            "ens":                      float(ens_stage2),
            "Weights_stage_2":      pd.Series(p)       

        },
        "predictor_stats": {
            "weighted": {
                col: {"mean": float(m), "std": float(s)}
                for col, m, s in zip(X_cols, weighted_mean, weighted_std)
            },
            "simple_vs_weighted": {
                col: {
                    "simple_mean":   float(sm),
                    "weighted_mean": float(wm),
                    "simple_std":    float(ss),
                    "weighted_std":  float(ws),
                }
                for col, sm, wm, ss, ws in zip(X_cols, simple_mean, weighted_mean, simple_std, weighted_std)
            },
            "standardisation_check": standardisation_check,
        },
        "rescaled_weights": {
            "sum":         float(p_rescaled.sum()),
            "most_recent": float(p_rescaled[-1]),
            "oldest":      float(p_rescaled[0]),
            "T":           int(T),
            "valid":       bool(abs(p_rescaled.sum() - T) < 1e-6),
        },
        "coefficients": {
            "standardised": {
                "intercept": float(intercept_std),
                "betas":     {col: float(b) for col, b in zip(X_cols, beta_std)},
            },
            "raw": {
                "intercept":    float(intercept_raw),
                "betas":        betas_raw_dict,
                "betas_ranked": betas_raw_sorted,
                "units":        "ALSI log return per unit change in predictor",
            },
        },
        "fit": {
            "weighted_r2":   float(r2_weighted),
            "unweighted_r2": float(r2_unweighted),
            "r2_gap":        float(r2_weighted - r2_unweighted),
            "overfit_flag":  bool((r2_weighted - r2_unweighted) > 0.20),
            "LOO_r2":     float(r2_loo)
        },
        'diagnostics':
                {
                    "jarque_Bera": jb_p,
                    'Durbin_Watson':dw
                }
    }

    # Realised per-observation weights for THIS window, date-indexed so the
    # frontend can plot them against the conditioning variable on a shared
    # time axis. p_rescaled = p·T (mean 1): values > 1 are upweighted dates.
    # The rolling fit keeps only the last (full-sample) window's copy.
    ridge_report["series"] = pd.Series(p_rescaled, index=df_Ridge_TimeFP.index)
    if conditioning_variable is not None:
        ridge_report["conditioning"] = {
            "variable":   conditioning_variable,
            "percentile": conditioning_percentile,
            "target":     float(target),                                  # the q-th percentile the tilt matches
            "series":     pd.Series(cond_values, index=df_Ridge_TimeFP.index),
        }
    else:
        ridge_report["conditioning"] = None

    return ridge_report

def rolling_fit_ridge_FP_SC(lamb, alpha,X_columns,y_column,conditioning_variable, conditioning_percentile,interaction_terms_list ,start_idx=250, keep_per_date_reports=False, ):
    X_columns = list(X_columns)  # FIX: do not mutate the caller's list (interaction cols were accumulating across calls)
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    for interaction_term_pair in interaction_terms_list:
        var1 = interaction_term_pair[0]
        var2 = interaction_term_pair[1]
        df[f'{var1}_x_{var2}'] = df[var1] * df[var2]
        X_columns.append(f'{var1}_x_{var2}')
    
    eval_dates = df.index[start_idx:]

    reports_per_date = {}
    infeasible_dates = []
    for date in eval_dates:
        df_slice = df.loc[:date].copy()
        report = fit_ridge_FP_SC(df_slice, lamb=lamb, alpha=alpha,
                                 X_cols=X_columns, y_col=y_column,
                                 conditioning_variable=conditioning_variable,
                                 conditioning_percentile=conditioning_percentile)
        if report.get('params', {}).get('conditioning_status') == 'INFEASIBLE':
            infeasible_dates.append(date)
            continue
        reports_per_date[date] = report

    state_conditioning_diagnostics = {
        'conditioning_variable':   conditioning_variable,
        'conditioning_percentile': conditioning_percentile,
        'attempted':               int(len(eval_dates)),
        'feasible':                int(len(reports_per_date)),
        'infeasible':              int(len(infeasible_dates)),
        'feasible_pct':            float(len(reports_per_date) / len(eval_dates)) if len(eval_dates) else 0.0,
        'infeasible_dates':        [str(d.date()) for d in infeasible_dates],
    }

    # No feasible fits — return 200 with diagnostics so the frontend can
    # render the "no feasible fits" panel instead of empty charts.
    if not reports_per_date:
        return {
            'data': {},
            'structured': {
                'params': {
                    'lambda':     float(lamb) if lamb is not None else None,
                    'alpha':      float(alpha),
                    'start_idx':  int(start_idx),
                    'num_fits':   0,
                    'eval_start': str(eval_dates[0].date()) if len(eval_dates) else None,
                    'eval_end':   str(eval_dates[-1].date()) if len(eval_dates) else None,
                    'y_col':      y_column,
                },
                'state_conditioning_diagnostics': state_conditioning_diagnostics,
                'no_feasible_fits': True,
            },
        }

    X_cols = list(next(iter(reports_per_date.values()))['coefficients']['raw']['betas'].keys())

    r2_weighted    = pd.Series({d: r['fit']['weighted_r2']   for d, r in reports_per_date.items()})
    r2_unweighted  = pd.Series({d: r['fit']['unweighted_r2'] for d, r in reports_per_date.items()})
    r2_gap         = r2_weighted - r2_unweighted
    overfit_flags  = pd.Series({d: r['fit']['overfit_flag']  for d, r in reports_per_date.items()})
    LOO_r2  = pd.Series({d: r['fit']['LOO_r2']  for d, r in reports_per_date.items()})
    jarque_beras  = pd.Series({d: r['diagnostics']['jarque_Bera']  for d, r in reports_per_date.items()})
    durbin_watsons  = pd.Series({d: r['diagnostics']['Durbin_Watson']  for d, r in reports_per_date.items()})
    
    betas_raw      = pd.DataFrame({d: r['coefficients']['raw']['betas']          for d, r in reports_per_date.items()}).T
    betas_std      = pd.DataFrame({d: r['coefficients']['standardised']['betas'] for d, r in reports_per_date.items()}).T
    intercepts_raw = pd.Series({d: r['coefficients']['raw']['intercept']          for d, r in reports_per_date.items()})
    intercepts_std = pd.Series({d: r['coefficients']['standardised']['intercept'] for d, r in reports_per_date.items()})


    # FIX: previous loop overwrote sign_flips/cv each iteration — only the last column was reported.
    beta_stability = {}
    for col in betas_raw.columns:
        s = betas_raw[col]
        mean_abs = abs(s.mean())
        beta_stability[col] = {
            'sign_flip_rate': float((np.diff(np.sign(s)) != 0).mean()),
            'cv': float(s.std() / mean_abs) if mean_abs > 1e-12 else None,  # CV undefined when the mean path ~0 (e.g. late ZARUSD)
        }

    
    ens = pd.Series({d: r['Stage2_diagnostics']['ens'] for d, r in reports_per_date.items()})
    
    data = {
        'fit': {
            'r2_weighted':   r2_weighted,
            'r2_unweighted': r2_unweighted,
            'r2_gap':        r2_gap,
            'overfit_flags': overfit_flags,
            'LOOCV_r2': LOO_r2
        },
        'coefficients': {
            'betas_raw':      betas_raw,
            'betas_std':      betas_std,
            'intercepts_raw': intercepts_raw,
            'intercepts_std': intercepts_std,
        },
        'weights_diagnostics': {
            'ens': ens,
            'jarque_beras' : jarque_beras,
            'durbin_watsons' : durbin_watsons

        },
    }

    # last_report — the final (full-sample) feasible fit's realised weights, so
    # the frontend can show how the weighting fell across history and, when
    # state-conditioned, line it up against the conditioning variable + target.
    last_date = max(reports_per_date.keys())
    last = reports_per_date[last_date]
    data['last_report'] = {
        'eval_date':    str(last_date.date()),
        'series':       last['series'],         # realised weights p·T, date-indexed
        'conditioning': last['conditioning'],   # None, or {variable, percentile, target, series}
    }

    if keep_per_date_reports:
        data['reports_per_date'] = reports_per_date
    
    feasible_dates_sorted = sorted(reports_per_date.keys())
    structured = {
        'params': {
            'lambda':     float(lamb) if lamb is not None else None,
            'alpha':      float(alpha),
            'start_idx':  int(start_idx),
            'num_fits':   len(reports_per_date),
            'eval_start': str(feasible_dates_sorted[0].date()),
            'eval_end':   str(feasible_dates_sorted[-1].date()),
        },
        'fit_summary': {
            'r2_weighted': {
                'mean':     float(r2_weighted.mean()),
                'std':      float(r2_weighted.std()),
                'min':      float(r2_weighted.min()),
                'max':      float(r2_weighted.max()),
                'min_date': str(r2_weighted.idxmin().date()),
                'max_date': str(r2_weighted.idxmax().date()),
            },
            'r2_unweighted': {
                'mean': float(r2_unweighted.mean()),
                'std':  float(r2_unweighted.std()),
                'min':  float(r2_unweighted.min()),
                'max':  float(r2_unweighted.max()),
            },
            'r2_gap': {
                'mean': float(r2_gap.mean()),
                'std':  float(r2_gap.std()),
                'min':  float(r2_gap.min()),
                'max':  float(r2_gap.max()),
            },
            'overfit_pct': float(overfit_flags.sum() / len(overfit_flags)),
            'CVLoo_r2':{
                'mean':     float(LOO_r2.mean()),
                'std':      float(LOO_r2.std()),
                'min':      float(LOO_r2.min()),
                'max':      float(LOO_r2.max()),
                'min_date': str(LOO_r2.idxmin().date()),
                'max_date': str(LOO_r2.idxmax().date()),
            }
        },
        'coefficient_summary': {
            col: {
                'mean':         float(betas_raw[col].mean()),
                'std':          float(betas_raw[col].std()),
                'min':          float(betas_raw[col].min()),
                'max':          float(betas_raw[col].max()),
                'start_value':  float(betas_raw[col].iloc[0]),
                'end_value':    float(betas_raw[col].iloc[-1]),
                'total_change': float(betas_raw[col].iloc[-1] - betas_raw[col].iloc[0]),
            }
            for col in X_cols
        },
        'weights_diagnostics_summary': {
            'ens': {
                'mean': float(ens.mean()),
                'min':  float(ens.min()),
                'max':  float(ens.max()),
            },
            'beta_stability': beta_stability,  # FIX: per-channel dict (was last-column scalar mislabelled as a max)
            'sign_flip_max': float(max(v['sign_flip_rate'] for v in beta_stability.values())),
            'CV_max': max((v['cv'] for v in beta_stability.values() if v['cv'] is not None), default=None),
        },
        'state_conditioning_diagnostics': state_conditioning_diagnostics,
        'no_feasible_fits': False,
    }

    return {'data': data, 'structured': structured}

# ============================================================================
# NEW — closes the gap between the methodology text and the code:
# (1) per-date ridge-penalty selection by out-of-sample (LOO) criterion;
# (2) drop-one incremental explained variance for correlated channel pairs.
# ============================================================================

def select_alpha_loo(df, lamb, X_cols, y_col, alpha_grid=None,
                     conditioning_variable=None, conditioning_percentile=None):
    """Return (alpha_star, table) where alpha_star maximises LOO R^2 on this
    slice over a log-spaced grid. Reuses fit_ridge_FP_SC so the selection
    criterion is exactly the reported out-of-sample metric — nothing bespoke."""
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 2, 41)  # 0.001 .. 100, 41 points (neighbours 1.33x apart, was 3.16x)
    table = {}
    for a in alpha_grid:
        rep = fit_ridge_FP_SC(df, lamb=lamb, alpha=float(a), X_cols=list(X_cols),
                              y_col=y_col, conditioning_variable=conditioning_variable,
                              conditioning_percentile=conditioning_percentile)
        if rep.get('params', {}).get('conditioning_status') == 'INFEASIBLE':
            continue
        table[float(a)] = float(rep['fit']['LOO_r2'])
    if not table:
        return None, table
    alpha_star = max(table, key=table.get)
    return alpha_star, table


def rolling_fit_ridge_FP_auto_alpha(lamb, X_columns, y_column,
                                    interaction_terms_list=None, start_idx=250,
                                    alpha_grid=None, reselect_every=13):
    """Rolling fit with the penalty selected by LOO R^2, re-selected every
    `reselect_every` estimation dates (default: quarterly on weekly data) to
    keep the alpha path smooth and the runtime sane. Returns the same
    structure as rolling_fit_ridge_FP_SC plus the alpha path."""
    X_columns = list(X_columns)
    interaction_terms_list = interaction_terms_list or []
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    for v1, v2 in interaction_terms_list:
        df[f'{v1}_x_{v2}'] = df[v1] * df[v2]
        X_columns.append(f'{v1}_x_{v2}')

    eval_dates = df.index[start_idx:]
    reports_per_date, alpha_path = {}, {}
    alpha_star = None
    for i, date in enumerate(eval_dates):
        df_slice = df.loc[:date].copy()
        if alpha_star is None or i % reselect_every == 0:
            alpha_star, _ = select_alpha_loo(df_slice, lamb, X_columns, y_column,
                                             alpha_grid=alpha_grid)
            if alpha_star is None:
                continue
        alpha_path[date] = alpha_star
        rep = fit_ridge_FP_SC(df_slice, lamb=lamb, alpha=alpha_star,
                              X_cols=X_columns, y_col=y_column,
                              conditioning_variable=None, conditioning_percentile=None)
        reports_per_date[date] = rep

    out = {
        'alpha_path': pd.Series(alpha_path),
        'r2_weighted':   pd.Series({d: r['fit']['weighted_r2']   for d, r in reports_per_date.items()}),
        'r2_unweighted': pd.Series({d: r['fit']['unweighted_r2'] for d, r in reports_per_date.items()}),
        'LOO_r2':        pd.Series({d: r['fit']['LOO_r2']        for d, r in reports_per_date.items()}),
        'betas_raw': pd.DataFrame({d: r['coefficients']['raw']['betas'] for d, r in reports_per_date.items()}).T,
        'ens':       pd.Series({d: r['Stage2_diagnostics']['ens'] for d, r in reports_per_date.items()}),
    }
    return out


def drop_one_delta_r2(df, lamb, alpha, X_cols, y_col):
    """Incremental explained variance per channel on this slice: full-model
    LOO R^2 minus the LOO R^2 with that channel removed, under identical
    weights. Well-defined even where individual betas are only partially
    identified (correlated pairs, e.g. USDZAR / yield differential) — this is
    the attribution reported jointly for such pairs in Section 4."""
    X_cols = list(X_cols)
    full = fit_ridge_FP_SC(df, lamb=lamb, alpha=alpha, X_cols=X_cols, y_col=y_col,
                           conditioning_variable=None, conditioning_percentile=None)
    r2_full = float(full['fit']['LOO_r2'])
    out = {'full_LOO_r2': r2_full, 'delta': {}}
    for col in X_cols:
        reduced_cols = [c for c in X_cols if c != col]
        red = fit_ridge_FP_SC(df, lamb=lamb, alpha=alpha, X_cols=reduced_cols, y_col=y_col,
                              conditioning_variable=None, conditioning_percentile=None)
        out['delta'][col] = r2_full - float(red['fit']['LOO_r2'])
    return out



    







# ============================ RUNNER ============================

import argparse, itertools, json, os, time, hashlib, warnings
warnings.filterwarnings('ignore')
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it
import numpy as np, pandas as pd

# ------------------------------------------------------------------ configuration
CSV = 'data/thesis_data.csv'
OUT = 'results'
ALPHA_GRID = np.logspace(-3, 2, 41); RESELECT = 4   # finer penalty grid, monthly re-selection (removes step artefacts in loading paths)
MAX_SECONDS = 0   # 0 = no budget; set e.g. 230 to run in sandbox-sized chunks and resume

TARGETS = {  # label -> dependent column
    'JALSH':  'JALSH Index_Log_Diff',
    'TOP40':  'TOP40 Index_Log_Diff',
    'RESI20': 'RESI20 Index_Log_Diff',
    'INDI25': 'INDI25 Index_Log_Diff',
    'FINI15': 'FINI15 Index_Log_Diff',
    'JMNNG':  'JMNNG Index_Log_Diff',   # JSE mining (confirm definition)
    'JGENR':  'JGENR Index_Log_Diff',   # JSE general retailers
    'MXZA':   'MXZA Index_Log_Diff',    # MSCI South Africa — NO SWIX break: control for the Mar-2024 natural experiment
    'MXZA_IN':'MXZA0IN Index_Log_Diff', # MSCI SA industrials
    'MXZA_CD':'MXZA0CD Index_Log_Diff', # MSCI SA consumer discretionary
    'MXZA_CS':'MXZA0CS Index_Log_Diff', # MSCI SA consumer staples
}
PRIMARY = 'JALSH'

F = {  # short name -> column   (ZAR Curncy excluded: identical to USDZAR; HG1 excluded: 0.95 with LME copper)
    'MM':    'MXWO0MM Index_Log_Diff',   # MSCI World Metals & Mining (contains BHP/AAL/GLEN -> JSE overlap)
    'MAT':   'MXWO0MT Index_Log_Diff',   # MSCI World Materials
    'USMM':  'S5METL Index_Log_Diff',    # S&P 500 Metals & Mining — CLEAN proxy, no JSE overlap
    'USMAT': 'S5MATRX Index_Log_Diff',   # S&P 500 Materials — clean proxy (confirm the X suffix)
    'CS':    'MXWO0CS Index_Log_Diff',   # World consumer staples
    'CD':    'MXWO0CD Index_Log_Diff',   # World consumer discretionary
    'FN':    'MXWO0FN Index_Log_Diff',   # World financials
    'USFN':  'S5FINL Index_Log_Diff',    # S&P 500 financials
    'USCS':  'S5CONS Index_Log_Diff',    # S&P 500 staples
    'USCD':  'S5COND Index_Log_Diff',    # S&P 500 discretionary
    'USCD':  'S5COND Index_Log_Diff',    # S&P 500 discretionary
    'EM':    'MXEF Index_Log_Diff',      # MSCI EM (contains SA ~3%)
    'WLD':   'MXWO Index_Log_Diff',      # MSCI World composite
    'VAL':   'MXWO000V Index_Log_Diff',  'GRO': 'MXWO000G Index_Log_Diff',  'QUA': 'M1WOQU Index_Log_Diff',
    'GOLD':  'XAU Curncy_Log_Diff',      'PLAT': 'XPT Curncy_Log_Diff',     'PALL': 'XPD Curncy_Log_Diff',
    'CU':    'LMCADS03 Comdty_Log_Diff', 'OIL': 'CL1 Comdty_Log_Diff',      'BCOM': 'BCOM Index_Log_Diff',
    'ZAR':   'USDZAR Curncy_Log_Diff',
    'ZARG':  'ZAR_glob_Log_Diff',          # rand, part spanned by global factors
    'ZARD':  'ZAR_dom_Log_Diff',           # rand, orthogonal remainder
    'DXY':   'DXY Curncy_Log_Diff',
    'VIX':   'VIX Index_Diff',           'US10': 'USGG10YR Index_Diff',
    'SPX':   'SPX Index_Log_Diff',       'SX5E': 'SX5E Index_Log_Diff',     'UKX': 'UKX Index_Log_Diff',
    # domestic panel only (thesis_data_domestic.csv) — runs on the main panel will KeyError if used
    'CDS':   'REPSOU CDS USD SR 5Y D14 Corp_Diff',
    'ALBIY': 'ALBIAY Index_Diff',
    'EMAF':  'M1FMEAFZ Index_Log_Diff',  # MSCI EM+Frontier Africa ex SA
}

FACTOR_POOL = ['MM','USMM','MAT','CS','EM','WLD','GOLD','PLAT','CU','OIL','ZAR','VIX']

CURATED = {
    # the model and its foils
    'baseline_mm':     ['MM','EM','CS','ZAR'],
    'baseline':        ['MAT','EM','CS','ZAR'],
    'monolithic':      ['WLD','ZAR'],
    'em_only':         ['EM'],
    'em_composite':    ['EM','ZAR'],
    'em_ladder_2': ['EM','ZAR','MM'],
    'pf':          ['WLD','ZAR'],              # Polakow and Flint's specification: one global factor and the rand
    'pf_em':       ['WLD','EM','ZAR'],         # their model with an emerging-market index added
    'pf_em_mm':    ['WLD','EM','ZAR','MM'],    # and with a mining channel added
    'split_rand':  ['EM','ZARG','ZARD','MM'],   # the presented model with the currency split in two
    'em_ladder_3': ['EM','ZAR','MM','CS'],
    'clean_ladder_2': ['EM','ZAR','USMM'],
    'clean_ladder_3': ['EM','ZAR','USMM','CS'],
    'ladder_1': ['WLD'], 'ladder_2': ['WLD','MM'], 'ladder_3': ['WLD','MM','EM'], 'ladder_4': ['WLD','MM','EM','CS'], 'ladder_5': ['WLD','MM','EM','CS','ZAR'],
    # circularity test: overlap-free mining proxies
    'baseline_clean':      ['USMM','EM','CS','ZAR'],
    'baseline_clean_mat':  ['USMAT','EM','CS','ZAR'],
    'baseline_clean_cs':   ['USMM','EM','USCS','ZAR'],
    'baseline_clean_stap': ['MM','EM','USCS','ZAR'],   # only the consumer channel swapped for the clean proxy
    'baseline_clean_disc': ['MM','EM','USCD','ZAR'],   # discretionary: Richemont/Prosus are discretionary, not staples
    'baseline_clean_stap': ['MM','EM','USCS','ZAR'],     # only the consumer proxy swapped
    'baseline_clean_disc': ['MM','EM','USCD','ZAR'],     # Richemont/Prosus are discretionary, not staples
    # extras that each test one thing
    'baseline_vix':    ['MM','EM','CS','ZAR','VIX'],
    'baseline_rates':  ['MM','EM','CS','ZAR','US10'],
    'baseline_fn':     ['MM','EM','CS','FN','ZAR'],
    'baseline_gold':   ['MM','EM','CS','ZAR','GOLD'],
    'baseline_cu':     ['MM','EM','CS','ZAR','CU'],
    'baseline_dxy':    ['MM','EM','CS','ZAR','DXY'],
    'style':           ['VAL','GRO','QUA','ZAR'],
    'commodity':       ['GOLD','PLAT','CU','OIL','ZAR'],
    'geographic':      ['SPX','SX5E','UKX','EM','ZAR'],
    'econscope':       ['WLD','CU','GOLD','PLAT','OIL','ZAR','DXY','VIX'],
}
# run with:  python run_specs.py --csv thesis_data_domestic.csv --full --specs ... --targets JALSH FINI15
DOMESTIC = {
    'baseline_cds':    ['MM','EM','CS','ZAR','CDS'],
    'baseline_albi':   ['MM','EM','CS','ZAR','ALBIY'],
    'baseline_emaf':   ['MM','EMAF','CS','ZAR'],     # Africa ex-SA in place of EM
    'baseline_mm_dom': ['MM','EM','CS','ZAR'],       # same spec on the domestic panel, for a like-for-like base
    'world_composite_dom': ['WLD','ZAR'],
    'em_composite_dom':    ['EM','ZAR'],
    'africa_composite':    ['EMAF','ZAR'],           # MSCI EM+Frontier Africa ex SA + rand
}

# ------------------------------------------------------------------ helpers
def build_synth(df):
    """Constructed columns, built at load time so the frozen data file is untouched.
    ZAR_glob / ZAR_dom: the rand's weekly move split into the part spanned by the global factors
    (fitted values of a full-sample regression on mining, EM, the dollar index, US 10-year yields,
    VIX and a commodity index) and the orthogonal remainder. Full-sample fitting is disclosed:
    the split is a decomposition for interpretation, not a forecasting device."""
    G = ['MXWO0MM Index_Log_Diff','MXEF Index_Log_Diff','DXY Curncy_Log_Diff',
         'USGG10YR Index_Diff','VIX Index_Diff','BCOM Index_Log_Diff']
    G = [g for g in G if g in df.columns]
    Z = 'USDZAR Curncy_Log_Diff'
    if Z in df.columns and len(G) >= 2:
        X = np.c_[np.ones(len(df)), df[G].values]
        b = np.linalg.lstsq(X, df[Z].values, rcond=None)[0]
        df['ZAR_glob_Log_Diff'] = X @ b
        df['ZAR_dom_Log_Diff'] = df[Z].values - df['ZAR_glob_Log_Diff']
        print(f"   synth: rand split, global share of variance {df['ZAR_glob_Log_Diff'].var()/df[Z].var():.2f}")
    return df

def load():
    df = pd.read_csv(globals()['CSV']); df['Date'] = pd.to_datetime(df['Date']); return build_synth(df.set_index('Date'))

def spec_id(short_names):
    return '+'.join(short_names)

def rolling(df, y_col, cols, tau, alpha, start_idx, step=1, alpha_grid=None, reselect_every=4):
    """Rolling fits every `step`-th estimation date. alpha='auto' selects by LOO on a grid,
    re-selected every `reselect_every` fitted dates. Returns dict of DataFrames/Series."""
    lamb = np.log(2) / tau
    dates = df.index[start_idx::step]
    B_raw, B_std, fit, contrib, a_star = {}, {}, {}, {}, None
    for i, date in enumerate(dates):
        sl = df.loc[:date]
        if alpha == 'auto':
            if a_star is None or i % reselect_every == 0:
                a_star, _ = select_alpha_loo(sl, lamb, list(cols), y_col, alpha_grid=alpha_grid)
                if a_star is None: continue
            a = a_star
        else:
            a = float(alpha)
        r = fit_ridge_FP_SC(sl, lamb=lamb, alpha=a, X_cols=list(cols), y_col=y_col,
                               conditioning_variable=None, conditioning_percentile=None)
        if r.get('params', {}).get('conditioning_status') == 'INFEASIBLE': continue
        braw = r['coefficients']['raw']['betas']; bstd = r['coefficients']['standardised']['betas']
        B_raw[date] = braw; B_std[date] = bstd
        fit[date] = {'r2_weighted': r['fit']['weighted_r2'], 'r2_unweighted': r['fit']['unweighted_r2'],
                     'LOO_r2': r['fit']['LOO_r2'], 'alpha': a, 'ens': r['Stage1_diagnostics']['ens'],
                     'intercept_raw': r['coefficients']['raw']['intercept']}
        x_t = df.loc[date, list(cols)]
        contrib[date] = {c: braw[c] * float(x_t[c]) for c in cols}   # signed weekly contribution, NOT aggregated
    return {'betas_raw': pd.DataFrame(B_raw).T, 'betas_std': pd.DataFrame(B_std).T,
            'fit': pd.DataFrame(fit).T, 'contrib_weekly': pd.DataFrame(contrib).T}

def save_run(res, target, sid, factors):
    d = os.path.join(OUT, target, sid); os.makedirs(d, exist_ok=True)
    for k, v in res.items(): v.to_csv(os.path.join(d, f'{k}.csv'))
    f = res['fit']
    return {'target': target, 'spec': sid, 'factors': factors, 'n_factors': len(factors), 'n_dates': len(f),
            'first_date': str(f.index.min().date()), 'last_date': str(f.index.max().date()),
            'LOO_r2_mean': f['LOO_r2'].mean(), 'r2_weighted_mean': f['r2_weighted'].mean(),
            'r2_unweighted_mean': f['r2_unweighted'].mean(), 'r2_gap_mean': (f['r2_weighted']-f['r2_unweighted']).mean(),
            'alpha_median': f['alpha'].median(), 'ens_last': f['ens'].iloc[-1],
            **{f'beta_end_{c}': res['betas_raw'][col].iloc[-1] for c, col in zip(factors, res['betas_raw'].columns)}}

# ------------------------------------------------------------------ tiers
def screen(df, tau, alpha, start_idx, step, max_k):
    y = TARGETS[PRIMARY]; rows = []
    pool = [c for c in FACTOR_POOL if F[c] in df.columns]
    subsets = [s for k in range(2, max_k+1) for s in itertools.combinations(pool, k)]
    part = os.path.join(OUT, 'screen_partial.csv'); os.makedirs(OUT, exist_ok=True)
    done = set(pd.read_csv(part)['spec']) if os.path.exists(part) else set()
    todo = [s for s in subsets if spec_id(s) not in done]
    print(f"SCREEN: {len(subsets)} subsets of {len(FACTOR_POOL)} factors, sizes 2..{max_k}, target {PRIMARY}, every {step}th date | {len(done)} done, {len(todo)} to go")
    t0 = time.time()
    pbar = tqdm(todo, desc=f"screen {PRIMARY}", unit="spec")
    for s in pbar:
        cols = [F[c] for c in s]
        res = rolling(df, y, cols, tau, alpha, start_idx, step=step, alpha_grid=ALPHA_GRID, reselect_every=RESELECT)
        f = res['fit']
        row = {'spec': spec_id(s), 'n_factors': len(s), 'LOO_r2_mean': f['LOO_r2'].mean(),
               'r2_weighted_mean': f['r2_weighted'].mean(), 'r2_unweighted_mean': f['r2_unweighted'].mean(), 'n_dates': len(f)}
        pd.DataFrame([row]).to_csv(part, mode='a', header=not os.path.exists(part), index=False)
        pbar.set_postfix(LOO=f"{row['LOO_r2_mean']:.3f}", spec=row['spec'])
        if MAX_SECONDS and time.time()-t0 > MAX_SECONDS: print("  time budget reached — rerun to resume"); break
    S = pd.read_csv(part).drop_duplicates('spec').sort_values('LOO_r2_mean', ascending=False).reset_index(drop=True)
    S.to_csv(os.path.join(OUT, 'screen.csv'), index=False)
    print(f"SCREEN done in {time.time()-t0:.0f}s. Top 10:\n{S.head(10).to_string(index=False)}")
    return S

def full(df, specs, targets, tau, alpha, start_idx):
    rows = []; t0 = time.time()
    jobs = [(t, sid, factors) for t in targets for sid, factors in specs.items()]
    pbar = tqdm(jobs, desc="full runs", unit="fit")
    for target, sid, factors in pbar:
        cols = [F[c] for c in factors]
        d = os.path.join(OUT, target, sid)
        if os.path.exists(os.path.join(d, 'fit.csv')):
            res = {k: pd.read_csv(os.path.join(d, f'{k}.csv'), index_col=0, parse_dates=True)
                   for k in ('betas_raw', 'betas_std', 'fit', 'contrib_weekly')}
            rows.append(save_run(res, target, sid, factors)); pbar.set_postfix(target=target, spec=sid, status="cached"); continue
        if MAX_SECONDS and time.time()-t0 > MAX_SECONDS: print("  time budget reached — rerun to resume"); break
        res = rolling(df, TARGETS[target], cols, tau, alpha, start_idx, alpha_grid=ALPHA_GRID, reselect_every=RESELECT)
        rows.append(save_run(res, target, sid, factors))
        pbar.set_postfix(target=target, spec=sid, LOO=f"{rows[-1]['LOO_r2_mean']:.3f}")
    # rebuild the summary from EVERY run on disk, not just this invocation's specs
    rows = []
    for target in os.listdir(OUT):
        tdir = os.path.join(OUT, target)
        if not os.path.isdir(tdir): continue
        for sid in os.listdir(tdir):
            d = os.path.join(tdir, sid)
            if not os.path.exists(os.path.join(d, 'fit.csv')): continue
            res = {k: pd.read_csv(os.path.join(d, f'{k}.csv'), index_col=0, parse_dates=True) for k in ('betas_raw','betas_std','fit','contrib_weekly')}
            factors = [k for k, col in F.items() if col in res['betas_raw'].columns] or list(res['betas_raw'].columns)
            rows.append(save_run(res, target, sid, factors))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    return S


def drop_one_path(df, spec_name, factors, target, tau, alpha, start_idx):
    """Per-date incremental explained variance for each factor in a spec:
    delta_k(t) = LOO(full)(t) - LOO(spec without k)(t), all under identical weights and penalty.
    Also writes the pairwise-joint version for every pair (drop both), which is the number
    reported for correlated pairs such as MM/EM."""
    y = TARGETS[target]; cols = [F[c] for c in factors]
    full = rolling(df, y, cols, tau, alpha, start_idx, alpha_grid=ALPHA_GRID, reselect_every=RESELECT)['fit']['LOO_r2']
    out = pd.DataFrame(index=full.index); out['LOO_full'] = full
    for k in factors:
        red = rolling(df, y, [F[c] for c in factors if c != k], tau, alpha, start_idx, alpha_grid=ALPHA_GRID, reselect_every=RESELECT)['fit']['LOO_r2']
        out[f'delta_{k}'] = full - red.reindex(full.index)
    for i, a in enumerate(factors):
        for b in factors[i+1:]:
            red = rolling(df, y, [F[c] for c in factors if c not in (a, b)], tau, alpha, start_idx, alpha_grid=ALPHA_GRID, reselect_every=RESELECT)['fit']['LOO_r2']
            out[f'delta_joint_{a}+{b}'] = full - red.reindex(full.index)
    d = os.path.join(OUT, target, spec_name); os.makedirs(d, exist_ok=True)
    out.to_csv(os.path.join(d, 'drop_one_path.csv'))
    summ = out.drop(columns='LOO_full').agg(['mean', 'min', 'max']).T
    summ['first_52w'] = out.drop(columns='LOO_full').iloc[:52].mean(); summ['last_52w'] = out.drop(columns='LOO_full').iloc[-52:].mean()
    summ.to_csv(os.path.join(d, 'drop_one_summary.csv'))
    print(f"\nDROP-ONE  {target} / {spec_name}  (mean LOO full {full.mean():.4f})"); print(summ.round(4).to_string())
    return out

# ------------------------------------------------------------------ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--screen', action='store_true'); ap.add_argument('--full', action='store_true')
    ap.add_argument('--tau', type=float, nargs='+', default=[104], help='half-life in weeks; pass several to sweep, e.g. --tau 52 118 156'); ap.add_argument('--alpha', default='auto')
    ap.add_argument('--step', type=int, default=26); ap.add_argument('--max-k', type=int, default=5)
    ap.add_argument('--top-k', type=int, default=8); ap.add_argument('--start-idx', type=int, default=250)
    ap.add_argument('--max-seconds', type=int, default=0)
    ap.add_argument('--csv', default=None, help='override data file')
    ap.add_argument('--out', default=None, help='override results folder')
    ap.add_argument('--drop-one', nargs='*', default=None, help='spec names to run drop-one delta-LOO paths for (default baseline_mm)')
    ap.add_argument('--grid-points', type=int, default=41)
    ap.add_argument('--reselect-every', type=int, default=4)
    ap.add_argument('--targets', nargs='*', default=list(TARGETS)); ap.add_argument('--specs', nargs='*', default=None)
    a = ap.parse_args()
    if a.csv: globals()['CSV'] = a.csv
    if a.out: globals()['OUT'] = a.out
    CSV = globals()['CSV']; OUT = globals()['OUT']; globals()['CSV_PATH'] = CSV; MAX_SECONDS = a.max_seconds
    BASE_OUT = OUT
    for TAU in a.tau:
      OUT = os.path.join(BASE_OUT, f'tau_{int(TAU)}')   # always nest by half-life, one tau or several
      print(f"\n==================== half-life tau = {TAU:.0f} weeks ({TAU/52:.2f} years)  ->  {OUT}/ ====================")
      ALPHA_GRID = np.logspace(-3, 2, a.grid_points); RESELECT = a.reselect_every
      df = load(); os.makedirs(OUT, exist_ok=True)
      meta = {'csv': CSV, 'sha256': hashlib.sha256(open(CSV,'rb').read()).hexdigest(), 'tau_weeks': TAU, 'tau_years': round(TAU/52, 3), 'alpha': a.alpha,
              'start_idx': a.start_idx, 'alpha_grid_points': a.grid_points, 'alpha_reselect_every': a.reselect_every, 'first_estimation_date': str(df.index[a.start_idx].date()), 'factors': F,
              'targets': TARGETS, 'curated': CURATED, 'pool': FACTOR_POOL}
      json.dump(meta, open(os.path.join(OUT, 'spec_registry.json'), 'w'), indent=2)

      specs = dict(CURATED); specs.update(DOMESTIC)
      have = set(df.columns)
      skipped = [k for k, v in specs.items() if any(F[c] not in have for c in v)]
      specs = {k: v for k, v in specs.items() if k not in skipped}
      if skipped: print(f"skipping {len(skipped)} specs whose columns are not in {CSV}: {skipped}")
      if a.screen:
          S = screen(df, TAU, a.alpha, a.start_idx, a.step, a.max_k)
          for sid in S['spec'].head(a.top_k):
              specs[f'screen_{sid}'] = sid.split('+')
      elif a.full and os.path.exists(os.path.join(OUT, 'screen.csv')):
          S = pd.read_csv(os.path.join(OUT, 'screen.csv'))
          for sid in S['spec'].head(a.top_k): specs[f'screen_{sid}'] = sid.split('+')
      if a.specs:
          missing = [k for k in a.specs if k not in specs]
          if missing: print(f'WARNING: requested specs not defined in this file: {missing}')
          specs = {k: v for k, v in specs.items() if k in a.specs}
      if a.drop_one is not None:
          for sid in (a.drop_one or ['baseline_mm']):
              for target in a.targets:
                  if sid in specs: drop_one_path(df, sid, specs[sid], target, TAU, a.alpha, a.start_idx)
      if a.full:
          print(f"FULL: {len(specs)} specs x {len(a.targets)} targets = {len(specs)*len(a.targets)} rolling fits")
          S = full(df, specs, a.targets, TAU, a.alpha, a.start_idx)
          if len(S): print("\nSUMMARY (JALSH):"); print(S[S.target=='JALSH'].sort_values('LOO_r2_mean', ascending=False)[['spec','n_factors','LOO_r2_mean','r2_weighted_mean','r2_unweighted_mean','alpha_median']].to_string(index=False))

    if True:
        frames = []
        for TAU in sorted({int(d.split('_')[1]) for d in os.listdir(BASE_OUT) if d.startswith('tau_') and os.path.isdir(os.path.join(BASE_OUT, d))}):
            p = os.path.join(BASE_OUT, f'tau_{int(TAU)}', 'summary.csv')
            if os.path.exists(p):
                s = pd.read_csv(p); s['tau_weeks'] = int(TAU); frames.append(s)
        if frames:
            A = pd.concat(frames)
            A.to_csv(os.path.join(BASE_OUT, 'summary_all_tau.csv'), index=False)
            piv = A.pivot_table(index=['target', 'spec'], columns='tau_weeks', values='LOO_r2_mean')
            piv.to_csv(os.path.join(BASE_OUT, 'tau_sensitivity_LOO.csv'))
            print("\nLOO R2 by half-life (JALSH):")
            _t = [t for t in piv.index.get_level_values(0).unique()]
    if _t: print(piv.loc[_t[0]].round(4).to_string())
