"""03_figures.py — thesis figures from results/. One function per figure; regenerates everything.
    python 03_figures.py            (expects results/ from 02_run_specs.py, tau folders or flat)
"""
import os, glob, numpy as np, pandas as pd, matplotlib
COLMAP = {'MM':'MXWO0MM Index_Log_Diff','EM':'MXEF Index_Log_Diff','ZAR':'USDZAR Curncy_Log_Diff','CS':'MXWO0CS Index_Log_Diff','USMM':'S5METL Index_Log_Diff'}
TARGET_COL = {'JALSH':'JALSH Index_Log_Diff','TOP40':'TOP40 Index_Log_Diff','RESI20':'RESI20 Index_Log_Diff','INDI25':'INDI25 Index_Log_Diff','FINI15':'FINI15 Index_Log_Diff','JMNNG':'JMNNG Index_Log_Diff','JGENR':'JGENR Index_Log_Diff','MXZA':'MXZA Index_Log_Diff'}
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'axes.edgecolor':'#00243A','axes.labelcolor':'#00243A','xtick.color':'#00243A','ytick.color':'#00243A','text.color':'#000000'})
RES = 'results/tau_104' if os.path.isdir('results/tau_104') else 'results'; OUT = 'figures'; os.makedirs(OUT, exist_ok=True)
PRIMARY = 'TOP40'   # the presented index
SPEC = 'em_ladder_2'; F = ['MM','EM','ZAR']          # presented model: three channels
SPEC4 = 'baseline_mm'                                  # four-channel candidate, used only in the 4.3 rejection and the 4.6 sector extension
# UCT palette: black 000000, blue 009ADA, dark blue 00243A; tints derived from the blue for the remaining channels.
UCT = {'black':'#000000','blue':'#009ADA','navy':'#00243A','light':'#7FCBEC','pale':'#BFE4F5','grey':'#B4B4B4','mid':'#4F6D7A',
       'orange':'#E87722','green':'#2E8B57'}
C = {'MM':UCT['navy'],'EM':UCT['blue'],'CS':UCT['green'],'ZAR':UCT['orange'],'unexplained':UCT['grey']}
LBL = {'MM':'Global metals & mining','EM':'Emerging markets','CS':'Global consumer staples','ZAR':'USDZAR'}
BREAK = pd.Timestamp('2024-03-18')
BREAK0 = pd.Timestamp('2023-06-19')   # Richemont treated as foreign for index purposes from the June 2023 review
NAME = {'JALSH':'All Share','TOP40':'Top 40','MXZA':'MSCI South Africa','RESI20':'Resources','JMNNG':'Mining','INDI25':'Industrials','FINI15':'Financials','JGENR':'Retailers'}
def nm(t): return NAME.get(t, t)
def load(t, s=SPEC):
    d = f'{RES}/{t}/{s}'; b = pd.read_csv(f'{d}/betas_raw.csv', index_col=0, parse_dates=True); bs = pd.read_csv(f'{d}/betas_std.csv', index_col=0, parse_dates=True)
    f = pd.read_csv(f'{d}/fit.csv', index_col=0, parse_dates=True); c = pd.read_csv(f'{d}/contrib_weekly.csv', index_col=0, parse_dates=True)
    ren = {v: k for k, v in COLMAP.items()}
    for x in (b, bs, c): x.rename(columns=ren, inplace=True)
    for x in (b, bs, c):
        missing = [k for k in x.columns if k not in C]
        if missing: print(f'   WARNING: unmapped columns in {t}/{s}: {missing}')
    return b, bs, f, c
def style(ax): ax.axvline(BREAK, color=UCT['mid'], ls=':', lw=1); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
def save(fig, name): fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png', dpi=200); plt.close(fig); print('  ', name)

def fig_loadings(t=PRIMARY):
    """Channel loadings through time, on their own."""
    b, bs, f, c = load(t)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k in [k for k in F if k in b.columns]: ax.plot(b.index, b[k], color=C[k], lw=1.5, label=LBL[k])
    ax.axhline(0, color=UCT['mid'], lw=.7); ax.set_ylabel('loading (raw units)')
    ax.legend(frameon=False, ncol=len(F), fontsize=8, loc='upper center', bbox_to_anchor=(0.5,-0.10))
    ax.set_title(f'{nm(t)}: channel loadings through time', loc='left', fontsize=10); style(ax)
    save(fig, f'fig_loadings_{t}')

def fig_r2_ens(t=PRIMARY):
    """The integration measure and the effective sample size, with the pandemic-step annotation."""
    b, bs, f, c = load(t)
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={'height_ratios':[2.4,1]})
    ax[0].plot(f.index, f.r2_weighted, color=UCT['navy'], lw=1.3, label='weighted R²')
    ax[0].plot(f.index, f.r2_unweighted, color=UCT['grey'], lw=1.1, label='unweighted R²')
    ax[0].plot(f.index, f.LOO_r2, color=UCT['black'], lw=1.3, label='leave-one-out R²')
    ax[0].set_ylabel('share of variance'); ax[0].legend(frameon=False, ncol=3, fontsize=8, loc='lower left')
    ax[0].set_title('(a) Integration measure', loc='left', fontsize=10)
    cv = pd.Timestamp('2020-03-20')
    if f.index.min() <= cv <= f.index.max():
        ax[0].annotate('step is leverage of the\npandemic weeks, not a change\nin exposure (Table 4)',
                       xy=(cv, float(f['LOO_r2'].asof(cv))), xytext=(pd.Timestamp('2015-01-01'), float(f.r2_weighted.max())),
                       textcoords='data', fontsize=7.5, color=UCT['mid'], ha='left', va='top',
                       arrowprops=dict(arrowstyle='->', color=UCT['mid'], lw=0.9))
    ax[1].plot(f.index, f.ens, color=UCT['black'], lw=1.2); ax[1].set_ylabel('ENS')
    ax[1].set_title('(b) Effective number of scenarios', loc='left', fontsize=10)
    for a in ax: style(a)
    save(fig, f'fig_r2_ens_{t}')

def fig_contrib_signed(t=PRIMARY, freq='QE'):
    b, bs, f, c = load(t); cm = c.resample(freq).sum(); fig, ax = plt.subplots(figsize=(11, 4.2)); w = 25 if freq=='ME' else 80
    bp = np.zeros(len(cm)); bn = np.zeros(len(cm))
    for k in [k for k in F if k in cm.columns]:
        p = cm[k].clip(lower=0); n = cm[k].clip(upper=0)
        ax.bar(cm.index, p, bottom=bp, width=w, color=C[k], label=LBL[k]); bp += p.values; ax.bar(cm.index, n, bottom=bn, width=w, color=C[k]); bn += n.values
    ax.plot(cm.index, cm.sum(axis=1), color='k', lw=.6, alpha=.6, label='total explained'); ax.axhline(0, color='k', lw=.6)
    ax.set_ylabel('contribution to log return'); ax.legend(frameon=False, ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12)); ax.set_title(f'Signed contributions β·x by channel, quarterly, {nm(t)}', loc='left', fontsize=10); style(ax)
    save(fig, f'fig_contrib_signed_{t}')

def fig_share_of_influence(t=PRIMARY):
    """Share of influence WITH residual for the presented model, quarterly, plus a leader strip."""
    b, bs, f, c = load(t)
    y = None
    for root in ('data/thesis_data.csv', 'thesis_data.csv'):
        if os.path.exists(root): y = pd.read_csv(root, index_col=0, parse_dates=True)[TARGET_COL.get(t, 'JALSH Index_Log_Diff')]; break
    if y is None: print('   share of influence: data file not found'); return
    resid = (y.reindex(c.index) - (f['intercept_raw'].reindex(c.index) + c.sum(axis=1))).abs().rename('unexplained')
    q = pd.concat([c.abs(), resid], axis=1).resample('QE').sum(); sh = q.div(q.sum(axis=1), axis=0)
    lead = sh[[k for k in F if k in sh.columns]].idxmax(axis=1)
    fig, (axl, ax) = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True, gridspec_kw={'height_ratios':[0.18, 4]})
    for dt, k in lead.items(): axl.bar(dt, 1, width=80, color=C[k])
    axl.set_yticks([]); axl.set_ylim(0, 1); axl.set_title('leader among the four channels', loc='left', fontsize=8, color=UCT['mid']); axl.spines[['top','right','left']].set_visible(False)
    bot = np.zeros(len(sh))
    for k in [k for k in F if k in sh.columns] + ['unexplained']:
        ax.bar(sh.index, sh[k]*100, bottom=bot, width=80, color=C[k], label=LBL.get(k, 'unexplained')); bot += sh[k].values*100
    ax.set_ylabel('% of |β·x| + |residual|'); ax.set_ylim(0, 100); ax.legend(frameon=False, ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f'Share of influence by channel, with residual, quarterly, {nm(t)}', loc='left', fontsize=10); style(ax); axl.axvline(BREAK, color=UCT['mid'], ls=':', lw=1)
    print('   share of influence means:', {k: round(float(sh[k].mean()),3) for k in F + ['unexplained']})
    save(fig, f'fig_share_of_influence_{t}')

def have(t, s=SPEC): return os.path.exists(f'{RES}/{t}/{s}/fit.csv')
def fig_subindex_validation(targets=('JALSH','TOP40','RESI20','JMNNG','INDI25','FINI15','JGENR')):
    targets = [t for t in targets if have(t)]
    M = pd.DataFrame({t: load(t)[1].mean() for t in targets}).T
    M = M[[k for k in F if k in M.columns]]; fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(M)); w = .2
    for i, k in enumerate(F): ax.bar(x+(i-1.5)*w, M[k], w, color=C[k], label=LBL[k])
    ax.set_xticks(x); ax.set_xticklabels([nm(t) for t in M.index], fontsize=8); ax.axhline(0, color='k', lw=.6); ax.set_ylabel('standardised loading (path mean)'); ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.set_title('Where each channel loads, by index', loc='left', fontsize=10); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.2)
    save(fig, 'fig_subindex_validation')

def fig_decomposition_gain(targets=('JALSH','TOP40','RESI20','JMNNG','INDI25','FINI15','JGENR')):
    S = pd.read_csv(f'{RES}/summary.csv'); piv = S[S.spec.isin(['monolithic', SPEC, 'clean_ladder_2'])].pivot(index='target', columns='spec', values='LOO_r2_mean'); piv = piv.loc[[t for t in targets if t in piv.index]]
    fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(piv))
    ax.bar(x-.27, piv.monolithic, .27, color=UCT['grey'], label='composite: MSCI World + USDZAR'); ax.bar(x, piv['clean_ladder_2'], .27, color=UCT['pale'], label='three channels, overlap-free mining proxy'); ax.bar(x+.27, piv[SPEC], .27, color=UCT['navy'], label='three channels, MSCI World M&M')
    ax.set_xticks(x); ax.set_xticklabels([nm(t) for t in piv.index], fontsize=8); ax.set_ylabel('mean leave-one-out R²'); ax.legend(frameon=False, fontsize=8); ax.set_title('Composite vs channel decomposition, by index', loc='left', fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.2); save(fig, 'fig_decomposition_gain')

def fig_ladder(t=PRIMARY):
    S = pd.read_csv(f'{RES}/summary.csv'); lad = S[(S.target==t) & S.spec.str.startswith('ladder')].set_index('spec').loc[[f'ladder_{i}' for i in range(1,6)], 'LOO_r2_mean']
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.plot(range(1,6), lad.values, marker='o', color=UCT['navy'])
    for i, v in enumerate(lad.values, 1): ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xticks(range(1,6)); ax.set_xticklabels(['World','+ Metals & Mining','+ EM','+ Staples','+ USDZAR'], fontsize=8); ax.set_ylabel('mean LOO R²'); ax.set_title(f'Nested ladder — {nm(t)}', loc='left', fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2); save(fig, f'fig_ladder_{t}')

def fig_em_ladder(t=PRIMARY):
    """The model built from the inherited specification: World + USDZAR, then EM + USDZAR, then with a
    mining channel. MSCI World Metals & Mining solid; the overlap-free S&P 500 Metals & Mining dashed."""
    def loo(spec):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/fit.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0)['LOO_r2'].mean()
        return None
    rungs = [loo('monolithic'), loo('em_composite'), loo('em_ladder_2')]
    clean = loo('clean_ladder_2')
    if any(v is None for v in rungs): print('   ladder: missing rungs'); return
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = [1, 2, 3]
    ax.plot(x, rungs, marker='o', color=UCT['navy'], lw=1.6, label='MSCI World Metals & Mining')
    for xi, v in zip(x, rungs): ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8, color=UCT['navy'])
    if clean is not None:
        ax.plot([2, 3], [rungs[1], clean], marker='s', color=UCT['orange'], lw=1.4, ls='--', label='S&P 500 Metals & Mining (no JSE constituents)')
        ax.annotate(f'{clean:.3f}', (3, clean), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, color=UCT['orange'])
    ax.set_xticks(x); ax.set_xticklabels(['MSCI World + USDZAR\n(inherited model)', 'MSCI EM + USDZAR', '+ mining channel'], fontsize=8.5)
    ax.set_ylabel('mean leave-one-out R²'); ax.set_xlim(0.7, 3.3)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title(f'Building the model from the inherited specification, {nm(t)}', loc='left', fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
    save(fig, f'fig_em_ladder_{t}')

def fig_em_ladder_full(t=PRIMARY):
    """The full ladder for 4.3: EM -> +USDZAR -> +mining -> +consumer staples, MSCI solid and overlap-free dashed."""
    def loo(spec):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/fit.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0)['LOO_r2'].mean()
        return None
    em, emz, mono = loo('em_only'), loo('em_composite'), loo('monolithic')
    msci = [em, emz, loo('em_ladder_2'), loo('em_ladder_3')]
    clean = [em, emz, loo('clean_ladder_2'), loo('clean_ladder_3')]
    if any(v is None for v in msci): print('   full ladder: missing MSCI rungs'); return
    fig, ax = plt.subplots(figsize=(7.4, 3.7)); x = [1, 2, 3, 4]
    ax.plot(x, msci, marker='o', color=UCT['navy'], lw=1.6, label='MSCI World Metals & Mining')
    for xi, v in zip(x, msci): ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8, color=UCT['navy'])
    if not any(v is None for v in clean[2:]):
        ax.plot(x[1:], clean[1:], marker='s', color=UCT['orange'], lw=1.4, ls='--', label='S&P 500 Metals & Mining (no JSE overlap)')
        for xi, v in zip(x[2:], clean[2:]): ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, color=UCT['orange'])
    if mono is not None:
        ax.axhline(mono, color=UCT['grey'], ls=':', lw=1); ax.text(1.0, mono, f'World + USDZAR  {mono:.3f}', va='bottom', ha='left', fontsize=7.5, color=UCT['mid'])
    ax.set_xticks(x); ax.set_xticklabels(['EM', '+ USDZAR', '+ mining channel', '+ consumer staples'], fontsize=8.5); ax.set_ylabel('mean LOO R²')
    ax.set_xlim(0.7, 4.4); ax.legend(frameon=False, fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.3))
    ax.set_title(f'The complete model, built from the EM index, {nm(t)}', loc='left', fontsize=10); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
    save(fig, f'fig_em_ladder_full_{t}')

def fig_aggregation_check(t=PRIMARY, base_spec='em_composite', added_spec='em_ladder_2', base_label='(a) EM + USDZAR',
                          added_label='(b) EM + USDZAR + Metals & Mining', tag='', title='before and after the mining channel enters'):
    """Share of influence WITH residual for a base spec and the same spec with one channel added."""
    def find(spec, k):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/{k}.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0, parse_dates=True)
        return None
    short = {'MXEF Index_Log_Diff':'EM','USDZAR Curncy_Log_Diff':'ZAR','MXWO0MM Index_Log_Diff':'MM','S5METL Index_Log_Diff':'MM','MXWO0CS Index_Log_Diff':'CS'}
    names = {'MM':'Global metals & mining','EM':'Emerging markets','ZAR':'USDZAR','CS':'Global consumer staples','unexplained':'unexplained'}
    if added_spec == 'clean_ladder_2': names['MM'] = 'S&P 500 Metals & Mining'
    y = None
    for root in ('data/thesis_data.csv', 'thesis_data.csv'):
        if os.path.exists(root): y = pd.read_csv(root, index_col=0, parse_dates=True)[TARGET_COL.get(t, 'JALSH Index_Log_Diff')]; break
    if y is None: print('   aggregation check: data file not found'); return
    panels = []
    for spec, ttl in [(base_spec, base_label), (added_spec, added_label)]:
        c, f = find(spec, 'contrib_weekly'), find(spec, 'fit')
        if c is None or f is None: print('   aggregation check: missing', spec); return
        panels.append((c.rename(columns=short), f, ttl))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True); stats = {}
    for a, (c, f, ttl) in zip(ax, panels):
        resid = (y.reindex(c.index) - (f['intercept_raw'].reindex(c.index) + c.sum(axis=1))).abs().rename('unexplained')
        q = pd.concat([c.abs(), resid], axis=1).resample('QE').sum(); sh = q.div(q.sum(axis=1), axis=0); bot = np.zeros(len(sh))
        order = [k for k in ['MM','EM','CS','ZAR'] if k in sh.columns] + ['unexplained']
        for k in order:
            a.bar(sh.index, sh[k]*100, bottom=bot, width=80, color=C.get(k, UCT['grey']), label=names[k]); bot += sh[k].values*100
        stats[ttl] = {k: float(sh[k].mean()) for k in order}
        a.set_ylim(0, 100); a.set_ylabel('% of |β·x| + |residual|'); a.set_title(f"{ttl}   (EM {sh['EM'].mean():.0%}, unexplained {sh['unexplained'].mean():.0%})", loc='left', fontsize=10); style(a)
    ax[1].legend(frameon=False, ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    fig.suptitle(f'Share of influence by channel, with residual, {title}, {nm(t)}', x=0.02, ha='left', fontsize=10)
    save(fig, f'fig_aggregation_check_{t}{tag}'); print('    ', {k: {kk: round(v,3) for kk,v in d.items()} for k,d in stats.items()}); return stats

def fig_aggregation_check_staples(t=PRIMARY):
    return fig_aggregation_check(t, base_spec='em_ladder_2', added_spec='em_ladder_3', base_label='(a) EM + USDZAR + Metals & Mining',
                                 added_label='(b) EM + USDZAR + Metals & Mining + Consumer Staples', tag='_staples', title='before and after the consumer channel enters')

def fig_aggregation_check_clean(t=PRIMARY):
    return fig_aggregation_check(t, base_spec='em_composite', added_spec='clean_ladder_2', added_label='(b) EM + USDZAR + S&P 500 Metals & Mining', tag='_clean')

def fig_staples_case(t=PRIMARY):
    """Out-of-sample fit with and without the consumer channel, through time, with the sample-mean gap marked."""
    def find(spec, k):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/{k}.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0, parse_dates=True)
        return None
    f3, f4 = find('em_ladder_2', 'fit'), find('em_ladder_3', 'fit')
    if f3 is None or f4 is None: print('   staples case: missing em_ladder_2 / em_ladder_3'); return
    gap = (f4['LOO_r2'] - f3['LOO_r2'].reindex(f4.index))
    print(f"   staples case: LOO without {f3['LOO_r2'].mean():.4f}, with {f4['LOO_r2'].mean():.4f}; gap first 52w {gap.iloc[:52].mean():+.4f}, last 52w {gap.iloc[-52:].mean():+.4f}, mean {gap.mean():+.4f}")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios':[2.2, 1]})
    ax[0].plot(f3.index, f3['LOO_r2'], color=UCT['mid'], lw=1.2, ls='--', label='without the consumer channel (EM + USDZAR + Metals & Mining)')
    ax[0].plot(f4.index, f4['LOO_r2'], color=C['CS'], lw=1.4, label='with the consumer channel')
    ax[0].fill_between(f4.index, f3['LOO_r2'].reindex(f4.index), f4['LOO_r2'], where=(gap>0), color=UCT['green'], alpha=.3, label='gain')
    ax[0].fill_between(f4.index, f3['LOO_r2'].reindex(f4.index), f4['LOO_r2'], where=(gap<0), color=UCT['orange'], alpha=.3, label='loss')
    ax[0].set_ylabel('leave-one-out R²'); ax[0].legend(frameon=False, fontsize=8, loc='lower left'); ax[0].set_title('(a) Out-of-sample fit with and without the consumer channel', loc='left', fontsize=10); style(ax[0])
    yr = gap.groupby(gap.index.year).mean()
    ax[1].bar(yr.index, yr.values, color=[C['CS'] if v>=0 else UCT['black'] for v in yr.values], width=0.8); ax[1].set_xticks(yr.index[::2])
    ax[1].axhline(gap.mean(), color=UCT['black'], ls='--', lw=1); ax[1].text(yr.index[0]-0.4, gap.mean(), f'  sample mean {gap.mean():+.3f}', va='bottom', fontsize=8, color=UCT['black'])
    ax[1].axhline(0, color=UCT['mid'], lw=.6); ax[1].set_ylabel('ΔLOO R²'); ax[1].set_title('(b) The gap, annual means', loc='left', fontsize=10)
    ax[1].spines[['top','right']].set_visible(False); ax[1].grid(axis='y', alpha=.2); ax[1].tick_params(axis='x', labelsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_staples_case_{t}.png', dpi=200); plt.close(fig); print('   ', f'fig_staples_case_{t}')

def fig_staples_by_sector(targets=('JALSH','TOP40','RESI20','JMNNG','INDI25','FINI15','JGENR')):
    """4.6 extension: LOO gain from adding the consumer channel to the presented model, by index."""
    def loo(t, spec):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/fit.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0)['LOO_r2'].mean()
        return None
    rows = [(t, loo(t, SPEC), loo(t, SPEC4)) for t in targets]
    rows = [(t, a, b) for t, a, b in rows if a is not None and b is not None]
    if not rows: print('   staples by sector: missing runs'); return
    lab = [nm(t) for t, _, _ in rows]; gain = [b - a for _, a, b in rows]
    fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(lab))
    ax.bar(x, [g*100 for g in gain], color=[UCT['green'] if g > 0.005 else UCT['grey'] for g in gain], width=.6)
    for xi, g in zip(x, gain): ax.annotate(f'{g*100:+.1f}', (xi, g*100), textcoords='offset points', xytext=(0, 4 if g>=0 else -12), ha='center', fontsize=8, color=UCT['mid'])
    ax.axhline(0, color=UCT['mid'], lw=.7); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylabel('percentage points of LOO R²')
    ax.set_title('What the consumer channel adds to the presented model, by index', loc='left', fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.2)
    print('   staples by sector:', {l: round(g*100,1) for l, g in zip(lab, gain)})
    save(fig, 'fig_staples_by_sector')

def fig_week_decomposition(t=PRIMARY, weeks=('2008-10-10','2020-03-13','2015-12-11','2025-04-04'), separate=True):
    """What the model claims for a single week: actual = intercept + channel contributions + residual.
    Emits one figure per week (fig_week_YYYY_MM_DD) and, if separate is False, a combined panel."""
    b, bs, f, c = load(t)
    y = None
    for root in ('data/thesis_data.csv', 'thesis_data.csv'):
        if os.path.exists(root): y = pd.read_csv(root, index_col=0, parse_dates=True)[TARGET_COL.get(t,'JALSH Index_Log_Diff')]; break
    if y is None: print('   week decomposition: data file not found'); return
    weeks = [pd.Timestamp(w) for w in weeks if pd.Timestamp(w) in c.index]
    if not weeks: print('   week decomposition: requested weeks not in the estimation range'); return

    def draw(ax, w, title_size=10):
        parts = [('intercept', float(f['intercept_raw'].loc[w]), UCT['grey'])]
        parts += [(LBL[k], float(c.loc[w, k]), C[k]) for k in F if k in c.columns]
        act = float(y.loc[w]); pred = sum(v for _, v, _ in parts); resid = act - pred
        parts.append(('residual', resid, UCT['mid']))
        def lab_at(ax, xi, v):
            lo, hi = ax.get_ylim(); below = (v*100 - 0.06*(hi-lo)) < lo
            ax.annotate(f'{v*100:+.1f}', (xi, v*100), textcoords='offset points',
                        xytext=(0, 4 if v >= 0 else (9 if below else -11)), ha='center', fontsize=8,
                        color='white' if below else UCT['mid'])
        for i, (lab, v, col) in enumerate(parts):
            ax.bar(i, v*100, color=col, width=.66); lab_at(ax, i, v)
        n = len(parts)
        ax.bar(n + 0.25, pred*100, color=UCT['mid'], width=.66, hatch='//', edgecolor='white', linewidth=0); lab_at(ax, n + 0.25, pred)
        ax.bar(n + 1.25, act*100, color=UCT['black'], width=.66); lab_at(ax, n + 1.25, act)
        ax.axvline(n - 0.4, color=UCT['grey'], lw=.8, ls=':')
        ax.axhline(0, color=UCT['mid'], lw=.7)
        ax.set_xticks(list(range(n)) + [n + 0.25, n + 1.25])
        ax.set_xticklabels([p[0] for p in parts] + ['model', 'actual'], rotation=40, ha='right', fontsize=8)
        ax.set_ylabel("% of the week's log return")
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.2)
        return act

    if separate:
        for w in weeks:
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            act = draw(ax, w)
            ax.set_title(f"{nm(t)}, week of {w.strftime('%d %B %Y')}: {act*100:+.1f}%", loc='left', fontsize=10)
            save(fig, f"fig_week_{w.strftime('%Y_%m_%d')}")
    n = len(weeks); rows = 2 if n > 2 else 1; cols = int(np.ceil(n/rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4*cols, 3.8*rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    titles = {'2008-10-10': '10 October 2008, global financial crisis', '2020-03-13': '13 March 2020, pandemic crash',
              '2025-04-04': '4 April 2025, tariff announcement', '2015-12-11': '11 December 2015, domestic political shock'}
    for ax, w in zip(axes, weeks):
        act = draw(ax, w)
        ax.set_title(titles.get(str(w.date()), f"week of {w.date()}") + f"   ({act*100:+.1f}%)", loc='left', fontsize=9.5)
        if ax not in list(axes[::cols]): ax.set_ylabel('')
    for ax in axes[len(weeks):]: ax.axis('off')
    lo, hi = axes[0].get_ylim(); axes[0].set_ylim(lo - 0.08*(hi-lo), hi)
    fig.suptitle(f'What the model claims for four weeks, {nm(t)}: components, the model total, and what happened', x=0.02, ha='left', fontsize=10)
    save(fig, f'fig_week_decomposition_{t}')

def fig_model_vs_actual(t=PRIMARY, weeks=('2008-10-10','2020-03-13','2025-04-04','2015-12-11'),
                        labels=('10 October 2008\nglobal financial crisis','13 March 2020\npandemic crash',
                                '4 April 2025\ntariff announcement','11 December 2015\ndomestic political shock')):
    """The verdict for each week, as a 2x2 panel: what the model predicted against what happened.
    Channel components are in the accompanying table; this figure carries only the comparison."""
    b, bs, f, c = load(t)
    y = None
    for root in ('data/thesis_data.csv', 'thesis_data.csv'):
        if os.path.exists(root): y = pd.read_csv(root, index_col=0, parse_dates=True)[TARGET_COL.get(t,'JALSH Index_Log_Diff')]; break
    if y is None: print('   model vs actual: data file not found'); return
    ws = [pd.Timestamp(w) for w in weeks if pd.Timestamp(w) in c.index]
    if len(ws) < 4: print('   model vs actual: need four weeks in range'); return
    vals = [(float(f['intercept_raw'].loc[w] + c.loc[w].sum())*100, float(y.loc[w])*100) for w in ws]
    lo = min(min(p, a) for p, a in vals); hi = max(max(p, a) for p, a in vals)
    pad = 0.18*(hi-lo); ylim = (lo-pad, hi+pad*0.6)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharey=True)
    for ax, w, lab, (p, a) in zip(axes.flat, ws, labels, vals):
        ax.bar(0, p, 0.55, color=UCT['mid'], hatch='//', edgecolor='white', linewidth=0)
        ax.bar(1, a, 0.55, color=UCT['black'])
        for xi, v in [(0, p), (1, a)]:
            ax.annotate(f'{v:+.1f}', (xi, v), textcoords='offset points', xytext=(0, 4 if v >= 0 else -12), ha='center', fontsize=9, color=UCT['mid'])
        ax.axhline(0, color=UCT['mid'], lw=.7); ax.set_xlim(-0.6, 1.6); ax.set_ylim(*ylim)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['model', 'actual'], fontsize=9)
        ax.set_title(lab, loc='left', fontsize=9.5)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.2)
    for ax in axes[:, 0]: ax.set_ylabel("% of the week's log return")
    fig.suptitle(f'What the model predicted and what happened, {nm(t)}', x=0.02, ha='left', fontsize=10)
    print('   model vs actual:', {str(w.date()): (round(p,1), round(a,1)) for w, (p, a) in zip(ws, vals)})
    save(fig, f'fig_model_vs_actual_{t}')

def fig_currency_taxonomy(t=None, targets=('RESI20','JMNNG','TOP40','JALSH','MXZA','INDI25','FINI15','JGENR'), smooth=26):
    """Flint's rand-hedge taxonomy made dynamic: the currency loading of each index through time,
    against the category boundaries his classification implies. Sectors migrate between categories."""
    NM = {'JALSH':'All Share','TOP40':'Top 40','MXZA':'MSCI South Africa','RESI20':'Resources 20',
          'JMNNG':'Mining','INDI25':'Industrials 25','FINI15':'Financials 15','JGENR':'General Retailers'}
    COL = {'RESI20':UCT['navy'],'JMNNG':'#1b4a63','TOP40':UCT['black'],'JALSH':UCT['mid'],
           'MXZA':UCT['grey'],'INDI25':UCT['green'],'FINI15':UCT['blue'],'JGENR':UCT['orange']}
    paths = {}
    for tg in targets:
        p = None
        for root in (RES, 'results', 'results/tau_104'):
            q = f'{root}/{tg}/{SPEC}/betas_raw.csv'
            if os.path.exists(q): p = q; break
        if p is None: continue
        b = pd.read_csv(p, index_col=0, parse_dates=True)
        ren = {v: k for k, v in COLMAP.items()}; b.rename(columns=ren, inplace=True)
        if 'ZAR' not in b.columns: continue
        paths[tg] = b['ZAR'].rolling(smooth, min_periods=1).mean()
    if not paths: print('   currency taxonomy: no runs found'); return
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bands = [(0.45, 2.0, 'rand hedge', '#e8f4fa'), (0.15, 0.45, 'partial hedge', '#f4fafd'),
             (-0.15, 0.15, 'no currency exposure', '#f7f7f7'), (-2.0, -0.15, 'rand play', '#fdf0e6')]
    for lo, hi, lab, c in bands:
        ax.axhspan(lo, hi, color=c, zorder=0)
        ax.annotate(lab, xy=(0.004, (max(lo, -0.95) + min(hi, 1.35)) / 2), xycoords=('axes fraction', 'data'),
                    fontsize=8, color=UCT['mid'], va='center', ha='left', zorder=1)
    for tg, s in paths.items():
        ax.plot(s.index, s.values, color=COL.get(tg, UCT['mid']), lw=1.5, label=NM.get(tg, tg), zorder=3)
        ax.annotate(NM.get(tg, tg), xy=(s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=COL.get(tg, UCT['mid']), va='center', zorder=3)
    for y in (0.45, 0.15, -0.15): ax.axhline(y, color='white', lw=1.1, zorder=2)
    ax.axhline(0, color=UCT['mid'], lw=.7, zorder=2)
    ax.axvline(BREAK, color=UCT['mid'], ls=':', lw=1, zorder=2)
    ax.set_ylabel('loading on USDZAR (raw units)'); ax.set_ylim(-1.0, 1.4)
    ax.set_xlim(min(s.index[0] for s in paths.values()), max(s.index[-1] for s in paths.values()) + pd.Timedelta(days=1500))
    ax.set_title('The currency exposure of the Johannesburg market through time, by index', loc='left', fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.15, zorder=1)
    print('   currency taxonomy:', {k: f'{v.iloc[:52].mean():+.2f} -> {v.iloc[-52:].mean():+.2f}' for k, v in paths.items()})
    save(fig, 'fig_currency_taxonomy')

def fig_pf_buildup(t=PRIMARY, panels=('pf','em_ladder_2')):
    """Share of influence with residual, quarterly, for three nested models starting from Polakow and
    Flint's specification: World + USDZAR; with MSCI EM added; with a mining channel added. Shows the
    literature's factor losing its share as channels are named."""
    def find(spec, k):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/{k}.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0, parse_dates=True)
        return None
    y = None
    for root in ('data/thesis_data.csv', 'thesis_data.csv'):
        if os.path.exists(root): y = pd.read_csv(root, index_col=0, parse_dates=True)[TARGET_COL.get(t,'JALSH Index_Log_Diff')]; break
    if y is None: print('   pf buildup: data file not found'); return
    short = {'MXWO Index_Log_Diff':'World','MXEF Index_Log_Diff':'EM','USDZAR Curncy_Log_Diff':'ZAR','MXWO0MM Index_Log_Diff':'MM'}
    colr = {'World':UCT['mid'],'EM':C['EM'],'ZAR':C['ZAR'],'MM':C['MM'],'unexplained':UCT['grey']}
    names = {'World':'MSCI World','EM':'MSCI Emerging Markets','ZAR':'USDZAR','MM':'MSCI World Metals & Mining','unexplained':'unexplained'}
    titles = {'pf':'MSCI World + USDZAR, the inherited specification','pf_em':'with MSCI Emerging Markets added','pf_em_mm':'with a mining channel added','em_ladder_2':'the presented model: MSCI EM, MSCI World Metals & Mining, USDZAR'}
    panels = [(p, f'({chr(97+i)}) {titles[p]}') for i, p in enumerate(panels)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.9*len(panels)+0.6), sharex=True); stats = {}
    axes = np.atleast_1d(axes)
    for ax, (spec, ttl) in zip(axes, panels):
        c, fi = find(spec,'contrib_weekly'), find(spec,'fit')
        if c is None or fi is None: print('   pf buildup: missing', spec); plt.close(fig); return
        c = c.rename(columns=short)
        resid = (y.reindex(c.index) - (fi['intercept_raw'].reindex(c.index) + c.sum(axis=1))).abs().rename('unexplained')
        q = pd.concat([c.abs(), resid], axis=1).resample('QE').sum(); sh = q.div(q.sum(axis=1), axis=0); bot = np.zeros(len(sh))
        order = [k for k in ['MM','EM','World','ZAR'] if k in sh.columns] + ['unexplained']
        for k in order:
            ax.bar(sh.index, sh[k]*100, bottom=bot, width=80, color=colr[k], label=names[k]); bot += sh[k].values*100
        stats[spec] = {k: round(float(sh[k].mean()),3) for k in order}
        ax.set_ylim(0,100); ax.set_ylabel('% of |β·x| + |residual|')
        ax.set_title(ttl + '   (' + ', '.join(f"{names[k].split()[-1] if k!='unexplained' else k} {sh[k].mean():.0%}" for k in order) + ')', loc='left', fontsize=9.5); style(ax)
    axes[-1].legend(frameon=False, ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5,-0.18))
    fig.suptitle(f'Share of influence by channel, with residual, inherited specification and presented model, {nm(t)}', x=0.02, ha='left', fontsize=10)
    print('   pf buildup:', stats)
    save(fig, f'fig_pf_buildup_{t}')

def fig_zar_loading_two(targets=('TOP40','MXZA'), smooth=13):
    """The USDZAR loading through time for the Top 40 and MSCI South Africa. The observation 4.2 opens on."""
    NMX = {'TOP40':'FTSE/JSE Top 40','MXZA':'MSCI South Africa'}
    COLX = {'TOP40':UCT['navy'],'MXZA':UCT['orange']}
    fig, ax = plt.subplots(figsize=(8.6, 4.0)); ok = False
    for t in targets:
        p = None
        for root in (RES, 'results', 'results/tau_104'):
            q = f'{root}/{t}/{SPEC}/betas_raw.csv'
            if os.path.exists(q): p = q; break
        if p is None: print('   zar loading: no run for', t); continue
        b = pd.read_csv(p, index_col=0, parse_dates=True); b.rename(columns={v: k for k, v in COLMAP.items()}, inplace=True)
        s = b['ZAR'].rolling(smooth, min_periods=1).mean(); ok = True
        ax.plot(s.index, s, color=COLX[t], lw=1.8, label=NMX[t])
        ax.annotate(f"{b['ZAR'].iloc[-52:].mean():+.2f}", xy=(s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8, color=COLX[t], va='center')
        ax.annotate(f"{b['ZAR'].iloc[:52].mean():+.2f}", xy=(s.index[0], s.iloc[0]), xytext=(-6, 0), textcoords='offset points', fontsize=8, color=COLX[t], va='center', ha='right')
    if not ok: plt.close(fig); return
    ax.axhline(0, color=UCT['mid'], lw=.8)
    ax.axvline(BREAK, color=UCT['grey'], ls=':', lw=1)
    ax.axvline(BREAK0, color=UCT['grey'], ls=':', lw=1)
    ax.set_ylabel('loading on USDZAR'); ax.set_xlim(ax.get_xlim()[0] - 300, ax.get_xlim()[1] + 500)
    ax.set_title('The loading on the rand, through time, with the two equity channels held fixed', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper right'); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
    save(fig, 'fig_zar_loading_two')

def fig_loading_reallocation(t=PRIMARY):
    """4.2: the EM loading with and without the mining channel, and the mining loading that takes the difference."""
    def find(spec):
        for root in (RES, 'results', 'results/tau_104'):
            p = f'{root}/{t}/{spec}/betas_raw.csv'
            if os.path.exists(p): return pd.read_csv(p, index_col=0, parse_dates=True)
        return None
    b2, b3 = find('em_composite'), find('em_ladder_2')
    if b2 is None or b3 is None: print('   loading reallocation: missing em_composite / em_ladder_2'); return
    EM='MXEF Index_Log_Diff'; MM='MXWO0MM Index_Log_Diff'
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(b2.index, b2[EM], color=C['EM'], lw=1.3, ls='--', label='EM loading, without a mining channel')
    ax.plot(b3.index, b3[EM], color=C['EM'], lw=1.6, label='EM loading, with a mining channel')
    ax.fill_between(b3.index, b2[EM].reindex(b3.index), b3[EM], color=C['EM'], alpha=.16, label='credit reassigned from EM')
    ax.plot(b3.index, b3[MM], color=C['MM'], lw=1.6, label='mining channel loading')
    ax.axhline(0, color=UCT['mid'], lw=.6); ax.set_ylabel('loading (raw units)')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5,-0.10))
    ax.set_title(f'What the mining channel takes from the EM composite, {nm(t)}', loc='left', fontsize=10); style(ax)
    print(f"   reallocation: EM without {b2[EM].mean():.3f}, with {b3[EM].mean():.3f}, MM {b3[MM].mean():.3f}")
    save(fig, f'fig_loading_reallocation_{t}')

def fig_tau_sensitivity():
    p = 'results/tau_sensitivity_LOO.csv'
    if not os.path.exists(p): return
    T = pd.read_csv(p); T = T[T.target=='JALSH'].set_index('spec').drop(columns='target'); rows = [s for s in ['monolithic','em_composite','baseline','baseline_clean',SPEC] if s in T.index]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for s in rows: ax.plot([int(c) for c in T.columns], T.loc[s].values, marker='o', label=s)
    ax.set_xlabel('half-life (weeks)'); ax.set_ylabel('mean LOO R², All Share'); ax.legend(frameon=False, fontsize=8); ax.set_title('Half-life sensitivity', loc='left', fontsize=10); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
    save(fig, 'fig_tau_sensitivity')

def fig_subindex_loading_paths(targets=(('JALSH','All Share'),('RESI20','Resources 20'),('INDI25','Industrials 25'),('FINI15','Financials 15'))):
    targets = [x for x in targets if have(x[0])]
    if len(targets) < 4: return
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, (t, name) in zip(axs.flat, targets):
        bb = load(t)[0]
        for k in F: ax.plot(bb.index, bb[k], color=C[k], lw=1.2, label=LBL[k])
        ax.axhline(0, color=UCT['mid'], lw=.6); ax.set_title(name, loc='left', fontsize=10); style(ax)
    axs[0,0].legend(frameon=False, ncol=2, fontsize=7.5); save(fig, 'fig_subindex_loading_paths')

def fig_alpha_overlay(t=PRIMARY):
    sp = SPEC if os.path.isdir(f'results_auto/{t}/{SPEC}') else SPEC4
    if not os.path.isdir(f'results_auto/{t}/{sp}'): return
    bf, _, ff, _ = load(t, sp); ba = pd.read_csv(f'results_auto/{t}/{sp}/betas_raw.csv', index_col=0, parse_dates=True); ba.columns = F
    fa = pd.read_csv(f'results_auto/{t}/{sp}/fit.csv', index_col=0, parse_dates=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={'height_ratios':[2.4,1]})
    for k in F: ax[0].plot(ba.index, ba[k], color=C[k], lw=1.1, alpha=.4); ax[0].plot(bf.index, bf[k], color=C[k], lw=1.5, label=LBL[k])
    ax[0].axhline(0, color=UCT['mid'], lw=.6); ax[0].set_ylabel('loading (raw units)'); ax[0].legend(frameon=False, ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.02))
    ax[0].set_title(f'(a) Loadings under fixed α = 5.6 (solid) and leave-one-out-selected α (faint) — {nm(t)}', loc='left', fontsize=10)
    ax[1].plot(fa.index, fa.LOO_r2, color=UCT['mid'], lw=1, label=f'LOO R², selected α (mean {fa.LOO_r2.mean():.4f})'); ax[1].plot(ff.index, ff.LOO_r2, color=UCT['black'], lw=1.2, label=f'LOO R², fixed α (mean {ff.LOO_r2.mean():.4f})')
    ax[1].set_ylabel('LOO R²'); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title('(b) Out-of-sample fit', loc='left', fontsize=10)
    for a in ax: style(a)
    save(fig, f'fig_alpha_overlay_{t}')

def fig_three_composites(t=PRIMARY, root='results_domestic'):
    S = {'MSCI World + USDZAR': ('world_composite_dom', UCT['mid']), 'MSCI EM + USDZAR': ('em_composite_dom', UCT['blue']),
         'MSCI EM & Frontier Africa ex SA + USDZAR': ('africa_composite', UCT['green'])}
    if not all(os.path.exists(f'{root}/{t}/{s}/fit.csv') for s, _ in S.values()): return
    fig, ax = plt.subplots(figsize=(10, 4))
    for lab, (s, c) in S.items():
        f = pd.read_csv(f'{root}/{t}/{s}/fit.csv', index_col=0, parse_dates=True)
        ax.plot(f.index, f.LOO_r2, color=c, lw=1.3, label=f'{lab}  (mean {f.LOO_r2.mean():.3f})')
    ax.set_ylabel('leave-one-out R²'); ax.legend(frameon=False, fontsize=8, loc='center left', bbox_to_anchor=(0.0, 0.42))
    ax.set_title(f'Three composites, one currency: out-of-sample explanatory power through time, {nm(t)}', loc='left', fontsize=10)
    style(ax); save(fig, f'fig_three_composites_{t}')

if __name__ == '__main__':
    print(f"reading {RES}/ -> writing {OUT}/")
    for fn in (fig_loadings, fig_r2_ens, fig_contrib_signed, fig_share_of_influence, fig_subindex_validation, fig_decomposition_gain, fig_ladder, fig_tau_sensitivity, fig_subindex_loading_paths, fig_alpha_overlay, fig_three_composites, fig_em_ladder, fig_em_ladder_full, fig_aggregation_check, fig_aggregation_check_clean, fig_aggregation_check_staples, fig_staples_case, fig_staples_by_sector, fig_week_decomposition, fig_model_vs_actual, fig_loading_reallocation, fig_currency_taxonomy, fig_pf_buildup, fig_zar_loading_two):
        try: fn()
        except Exception as e: print(f'   skipped {fn.__name__}: {e}')
    print("done")
