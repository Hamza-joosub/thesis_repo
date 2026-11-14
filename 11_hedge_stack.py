"""11_hedge_stack.py — the two hedge groups stacked: the total holds, the composition inverts.
Reads top40_weights_monthly.csv (Bloomberg month-end index weights, classified as in Table 2).
Default is semi-annual (April and October); --monthly draws every month-end."""
import sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
UCT={'navy':'#00243A','orange':'#E87722','mid':'#4F6D7A','grey':'#B4B4B4'}
ap=argparse.ArgumentParser(); ap.add_argument('weights',nargs='?',default='top40_weights_monthly.csv')
ap.add_argument('--monthly',action='store_true'); ap.add_argument('--figdir',default='figures'); a=ap.parse_args()
C=pd.read_csv(a.weights,index_col=0,parse_dates=True)
A=C if a.monthly else C[C.index.month.isin([4,10])]
A=A.copy(); A['tot']=A.dual+A.earn
x=np.arange(len(A)); w=0.85
fig,ax=plt.subplots(figsize=(10.5,4.4))
ax.bar(x,A.dual,color=UCT['navy'],label='dual-listed',width=w)
ax.bar(x,A.earn,bottom=A.dual,color=UCT['orange'],label='South African dollar-earners',width=w)
# label the two matched snapshots and the harmonisation
for dt,lab in [('2015-04-30','48'),('2026-10-31','49')]:
    if dt in A.index.strftime('%Y-%m-%d'):
        i=list(A.index.strftime('%Y-%m-%d')).index(dt)
        ax.annotate(f'{A.tot.iloc[i]:.0f}',(i,A.tot.iloc[i]),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8,color=UCT['mid'],fontweight='bold')
        ax.annotate(f'{A.dual.iloc[i]:.0f}',(i,A.dual.iloc[i]/2),ha='center',va='center',fontsize=7.5,color='white')
        ax.annotate(f'{A.earn.iloc[i]:.0f}',(i,A.dual.iloc[i]+A.earn.iloc[i]/2),ha='center',va='center',fontsize=7.5,color='white')
hi=[i for i,d in enumerate(A.index) if d>=pd.Timestamp('2024-03-18')]
if hi: ax.axvline(hi[0]-0.5,color=UCT['grey'],ls=':',lw=1.2); ax.annotate('free-float\nharmonisation\nMar 2024',xy=(hi[0]-0.5,A.tot.max()+2),fontsize=7,color=UCT['mid'],ha='center',va='top')
ax.annotate('same total weight in the two groups,\ninverted composition, eleven years apart',xy=(len(A)*0.45,A.tot.max()+6),fontsize=8,color=UCT['mid'],ha='center')
yrs=[i for i,d in enumerate(A.index) if d.month==(1 if a.monthly else 4)]
ax.set_xticks(yrs); ax.set_xticklabels([A.index[i].year for i in yrs],fontsize=7.5,rotation=45)
ax.set_ylim(0,A.tot.max()+22); ax.set_xlim(-0.6,len(A)-0.4); ax.set_ylabel('% of the Top 40')
ax.set_title(f"Weight of the two hedge groups in the Top 40, {'monthly' if a.monthly else 'April and October of each year'}",loc='left',fontsize=10)
ax.legend(frameon=False,fontsize=8,loc='upper left',ncol=2); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
import os; os.makedirs(a.figdir,exist_ok=True)
fig.tight_layout(); fig.savefig(f'{a.figdir}/fig_hedge_stack.png',dpi=200)
print(f'{len(A)} bars | {a.figdir}/fig_hedge_stack.png')
