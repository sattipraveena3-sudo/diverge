from dataclasses import dataclass,asdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
@dataclass
class Metrics:
    precision:float; recall:float; f1:float; alert_time_s:float|None; lead_time_s:float|None
    def to_dict(self)->dict: return asdict(self)
def first_alert_time(time_s,risk,min_consecutive:int=3):
    risk=np.asarray(risk,dtype=bool); run=0
    if min_consecutive<=0: raise ValueError("min_consecutive must be positive")
    for idx,flag in enumerate(risk):
        run=run+1 if flag else 0
        if run>=min_consecutive: return float(time_s[idx-min_consecutive+1])
    return None
def evaluate_predictions(time_s,y_true,risk,event_start_s):
    p,r,f1,_=precision_recall_fscore_support(y_true,risk.astype(int),average="binary",zero_division=0); alert=first_alert_time(time_s,risk); lead=None if alert is None or event_start_s is None else float(event_start_s-alert); return Metrics(float(p),float(r),float(f1),alert,lead)
