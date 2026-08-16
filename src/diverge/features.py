from __future__ import annotations
import pandas as pd
from .data import align_and_impute,robust_normalize
from .divergence import rolling_cross_correlation_divergence,rolling_residual_divergence
def build_features(frame:pd.DataFrame,window:int=60)->pd.DataFrame:
    work=robust_normalize(align_and_impute(frame)); work["corr_div"]=rolling_cross_correlation_divergence(work["hr_z"],work["spo2_z"],window); work["resid_div"]=rolling_residual_divergence(work["hr_z"],work["spo2_z"],window); work["hr_slope"]=work["hr_z"].diff(10).fillna(0)/10; work["spo2_slope"]=work["spo2_z"].diff(10).fillna(0)/10; work["divergence"]=0.6*work["corr_div"]+0.4*work["resid_div"]; return work
