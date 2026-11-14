"""fig_weight_shift.py — the two hedge groups' index weights, head to head."""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
UCT={'navy':'#00243A','orange':'#E87722','mid':'#4F6D7A','grey':'#B4B4B4'}
m=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'members_classified.csv')
rows=[]
for yr,g in m.groupby('Year'):
    t=g.nlargest(40,'W'); tot=t.W.sum()
    rows.append({'year':int(yr),
                 'dual':100*t[t.Category=='Offshore-priced'].W.sum()/tot,
                 'earn':100*t[t.Category=='Dollar-earning SA'].W.sum()/tot})
C=pd.DataFrame(rows).set_index('year').sort_index()
fig,ax=plt.subplots(figsize=(8.6,4.2))
ax.plot(C.index,C.dual,color=UCT['navy'],lw=2.2,marker='o',ms=4,label='dual-listed')
ax.plot(C.index,C.earn,color=UCT['orange'],lw=2.2,marker='^',ms=4,label='South African dollar-earners')
ax.fill_between(C.index,C.dual,C.earn,where=C.dual>=C.earn,color=UCT['navy'],alpha=.07)
ax.fill_between(C.index,C.dual,C.earn,where=C.dual<C.earn,color=UCT['orange'],alpha=.10)
ax.axvline(2024,color=UCT['grey'],ls=':',lw=1); ax.annotate('free-float\nharmonisation\nMar 2024',xy=(2023.9,36),fontsize=7,color=UCT['mid'],ha='right',va='top')
ax.axvline(2025.4,color=UCT['grey'],ls=':',lw=1); ax.annotate('gold and\nplatinum run',xy=(2025.5,12),fontsize=7,color=UCT['mid'],ha='left',va='top')
for yr in (2008,2015,2020,2026):
    if yr in C.index:
        ax.annotate(f'{C.dual[yr]:.0f}',(yr,C.dual[yr]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7.5,color=UCT['navy'])
        ax.annotate(f'{C.earn[yr]:.0f}',(yr,C.earn[yr]),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=7.5,color=UCT['orange'])
ax.set_ylim(0,55); ax.set_ylabel('% of the Top 40'); ax.set_xticks(list(range(2008,2027,2)))
ax.set_title('The two hedge groups, head to head: index weight at April of each year',loc='left',fontsize=10)
ax.legend(frameon=False,fontsize=8,loc='center right'); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig('figures/fig_weight_shift.png',dpi=200)
d20,e20=C.dual[2020],C.earn[2020]; d26,e26=C.dual[2026],C.earn[2026]
print(f'2020: dual {d20:.1f} earners {e20:.1f} | 2026: dual {d26:.1f} earners {e26:.1f}')
print(f'change: dual {d26-d20:+.1f} earners {e26-e20:+.1f}')
print('figures/fig_weight_shift.png')
