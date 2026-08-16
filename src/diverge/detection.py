from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
FEATURE_COLUMNS=["hr_z","spo2_z","hr_slope","spo2_slope","corr_div","resid_div"]
@dataclass
class DetectionResult:
    risk: np.ndarray
    confidence: np.ndarray

def future_event_target(frame:pd.DataFrame,horizon_s:int=120)->np.ndarray:
    event=frame.get("event",pd.Series(np.zeros(len(frame),dtype=int))).to_numpy().astype(int); y=np.zeros(len(event),dtype=int); positives=np.where(event==1)[0]
    if len(positives): y[max(0,positives[0]-horizon_s):]=1
    return y

def train_detector(frame:pd.DataFrame)->Pipeline:
    y=future_event_target(frame)
    if len(np.unique(y))<2: raise ValueError("training data requires both positive and negative target windows")
    model=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))]); model.fit(frame[FEATURE_COLUMNS],y); return model

def predict_detector(model:Pipeline,frame:pd.DataFrame,threshold:float=0.5)->DetectionResult:
    confidence=model.predict_proba(frame[FEATURE_COLUMNS])[:,1]; return DetectionResult(confidence>=threshold,confidence)

def threshold_baseline(frame:pd.DataFrame)->DetectionResult:
    risk=(frame["hr"]>=100)|(frame["spo2"]<=92); hr_score=((frame["hr"]-90)/20).clip(0,1); spo2_score=((95-frame["spo2"])/6).clip(0,1); confidence=np.maximum(hr_score,spo2_score).to_numpy(); return DetectionResult(risk.to_numpy(),confidence)
