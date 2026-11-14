"""fig_dual_listed_scatter.py — the dual-listed basket's response to the rand, before and after the
equity channels are removed. Left: as measured against USDZAR alone. Right: with MSCI EM and MSCI
World Metals & Mining partialled out of both sides (Frisch-Waugh-Lovell), so the slope is the
basket's loading in the presented model."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
UCT={'navy':'#00243A','blue':'#009ADA','orange':'#E87722','mid':'#4F6D7A','grey':'#B4B4B4'}
GROUP=sys.argv[2] if len(sys.argv)>2 else 'dual'
SETS={'dual':(['AGL.JO','GLN.JO','BHG.JO','S32.JO','CFR.JO','BTI.JO','PRX.JO','MNP.JO'],'dual-listed'),
      'earner':(['GFI.JO','HAR.JO','IMP.JO','SOL.JO','SSW.JO','KIO.JO','EXX.JO','NPH.JO'],'dollar-earner')}
DL,GLAB=SETS[GROUP]
panel=sys.argv[1] if len(sys.argv)>1 else 'data/thesis_data.csv'

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

px,SRC=load_prices(DL,'2003-10-01'); print('price source:',SRC)
R=np.log(px.resample('W-FRI').last()).diff()
P=pd.read_csv(panel,index_col=0,parse_dates=True)
ctl=P[['USDZAR Curncy_Log_Diff','MXEF Index_Log_Diff','MXWO0MM Index_Log_Diff']].reindex(R.index,method='nearest'); ctl.columns=['z','em','mm']
cols=[c for c in DL if c in R.columns]
bk=R[cols].mean(axis=1,skipna=True).where(R[cols].notna().sum(axis=1)>=3)
d=pd.concat([bk,ctl],axis=1).dropna(); d.columns=['y','z','em','mm']
X=np.c_[np.ones(len(d)),d[['em','mm']].values]
ry=d.y.values-X@np.linalg.lstsq(X,d.y.values,rcond=None)[0]
rz=d.z.values-X@np.linalg.lstsq(X,d.z.values,rcond=None)[0]
fig,axes=plt.subplots(1,2,figsize=(10.4,5.0),sharey=True)
for ax,(xx,yy,ttl,col,xl) in zip(axes,[
        (d.z.values,d.y.values,'(a) against the rand alone',UCT['orange'],'USDZAR weekly return, %'),
        (rz,ry,'(b) with the two equity channels removed from both',UCT['navy'],'USDZAR weekly return, %, equity channels removed')]):
    k=(np.abs(xx)<.08)&(np.abs(yy)<.15); xx,yy=xx[k],yy[k]
    b=np.polyfit(xx,yy,1); r2=np.corrcoef(xx,yy)[0,1]**2
    ax.scatter(100*xx,100*yy,s=5,alpha=.28,color=col,edgecolors='none')
    xs=np.array([-8.,8.])
    ax.plot(xs,100*b[1]+b[0]*xs,color=UCT['navy'],lw=2,label=f'fitted slope {b[0]:+.2f}')
    ax.plot(xs,xs,color=UCT['mid'],lw=1.1,ls='--',label='one for one')
    ax.axhline(0,color=UCT['grey'],lw=.6); ax.axvline(0,color=UCT['grey'],lw=.6)
    ax.set_xlim(-8,8); ax.set_ylim(-15,15); ax.set_xlabel(xl)
    ax.set_title(f'{ttl}\nslope {b[0]:+.2f}, R\u00b2 {r2:.2f}',loc='left',fontsize=9.5)
    ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
axes[0].set_ylabel(f'{GLAB} basket, weekly return, %')
fig.suptitle(f'The {GLAB} basket and the rand, before and after the equity channels are removed',x=0.01,ha='left',fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.93]); tag='dual_listed' if GROUP=='dual' else 'dollar_earner'
fig.savefig(f'figures/fig_{tag}_scatter.png',dpi=200)
print(f'figures/fig_{tag}_scatter.png')
