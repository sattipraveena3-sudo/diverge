from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

def rolling_cross_correlation_divergence(x,y,window:int=60)->np.ndarray:
    if window<3: raise ValueError("window must be >= 3")
    xs=pd.Series(np.asarray(x,dtype=float)); ys=pd.Series(np.asarray(y,dtype=float)); corr=xs.rolling(window,min_periods=max(3,window//3)).corr(ys); return (1.0-corr.abs()).clip(0,1).fillna(0.0).to_numpy()

def rolling_residual_divergence(x,y,window:int=60)->np.ndarray:
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); out=np.zeros_like(x,dtype=float)
    for i in range(len(x)):
        start=max(0,i-window+1); xx,yy=x[start:i+1],y[start:i+1]
        if len(xx)<4 or np.std(xx)<1e-9: continue
        slope,intercept=np.polyfit(xx,yy,1); resid=yy-(slope*xx+intercept); scale=np.median(np.abs(resid-np.median(resid)))*1.4826+1e-6; out[i]=min(abs(resid[-1])/(4.0*scale),1.0)
    return out

def dtw_distance(x:np.ndarray,y:np.ndarray)->float:
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if not len(x) or not len(y): raise ValueError("DTW inputs cannot be empty")
    cost=np.full((len(x)+1,len(y)+1),np.inf); cost[0,0]=0.0
    for i in range(1,len(x)+1):
        for j in range(1,len(y)+1):
            d=euclidean([x[i-1]],[y[j-1]]); cost[i,j]=d+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
    return float(cost[-1,-1]/(len(x)+len(y)))

def rolling_dtw_divergence(x,y,window:int=30,stride:int=5)->np.ndarray:
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); out=np.zeros(len(x))
    for i in range(window-1,len(x),stride):
        a=x[i-window+1:i+1]; b=y[i-window+1:i+1]; a=(a-a.mean())/(a.std()+1e-6); b=(b-b.mean())/(b.std()+1e-6); out[i]=min(dtw_distance(a,b)/2.0,1.0)
    return pd.Series(out).replace(0,np.nan).ffill().fillna(0).to_numpy()
