from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diverge.data import generate_synthetic_record
from diverge.detection import predict_detector, train_detector
from diverge.features import build_features

app = FastAPI(title="DiVerge API", version="0.1.0")


class AnalyzeRequest(BaseModel):
    time_s: list[float]
    hr: list[float]
    spo2: list[float]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    divergence_score: float
    risk_flag: bool
    confidence: float


@lru_cache(maxsize=1)
def get_model():
    return train_detector(build_features(generate_synthetic_record(seed=42).frame))


@app.get("/health")
def health():
    return {"status": "ok", "service": "diverge"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    n = len(payload.time_s)
    if n < 10 or not (len(payload.hr) == len(payload.spo2) == n):
        raise HTTPException(422, "time_s, hr and spo2 must have equal length >= 10")
    features = build_features(
        pd.DataFrame({"time_s": payload.time_s, "hr": payload.hr, "spo2": payload.spo2})
    )
    pred = predict_detector(get_model(), features, threshold=payload.threshold)
    return AnalyzeResponse(
        divergence_score=float(features["divergence"].iloc[-1]),
        risk_flag=bool(pred.risk[-1]),
        confidence=float(pred.confidence[-1]),
    )
