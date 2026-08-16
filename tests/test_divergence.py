import numpy as np
from diverge.divergence import rolling_cross_correlation_divergence,rolling_residual_divergence,dtw_distance
def test_cross_correlation_low_for_correlated_signals():
    x=np.sin(np.linspace(0,20,300)); y=-2*x; d=rolling_cross_correlation_divergence(x,y,window=50); assert d[-1]<0.05
def test_residual_divergence_rises_on_outlier():
    x=np.linspace(-2,2,200); y=2*x; y[-1]+=10; d=rolling_residual_divergence(x,y,window=60); assert d[-1]>0.5
def test_dtw_identical_is_zero():
    x=np.array([1.0,2.0,3.0]); assert dtw_distance(x,x)==0.0
