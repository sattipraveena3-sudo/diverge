from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diverge.data import generate_synthetic_record
from diverge.detection import predict_detector, train_detector
from diverge.features import build_features, build_pair_features

app = FastAPI(
    title="DiVerge API",
    version="0.2.0",
    description=(
        "Research API for sensor-agnostic physiological relational divergence. "
        "Not a medical device and not for clinical decision-making."
    ),
)


class AnalyzeRequest(BaseModel):
    time_s: list[float]
    hr: list[float]
    spo2: list[float]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    divergence_score: float
    risk_flag: bool
    confidence: float


class PairAnalyzeRequest(BaseModel):
    time_s: list[float]
    signal_a: list[float | None]
    signal_b: list[float | None]
    signal_a_name: str = "signal_a"
    signal_b_name: str = "signal_b"
    divergence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    target_hz: float = Field(default=1.0, gt=0.0, le=20.0)


class PairAnalyzeResponse(BaseModel):
    signal_a_name: str
    signal_b_name: str
    samples_analyzed: int
    divergence_score: float
    divergence_velocity: float
    divergence_acceleration: float
    risk_flag: bool
    interpretation: str


@lru_cache(maxsize=1)
def get_model():
    """Synthetic demonstration model retained for backward-compatible /analyze."""
    return train_detector(build_features(generate_synthetic_record(seed=42).frame))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "diverge",
        "scope": "sensor-agnostic relational divergence research",
        "clinical_use": False,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    """Backward-compatible HR/SpO2 synthetic-demonstration endpoint."""
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


@app.post("/analyze-pair", response_model=PairAnalyzeResponse)
def analyze_pair(payload: PairAnalyzeRequest):
    """Analyze any synchronized physiological signal pair without domain assumptions.

    This endpoint intentionally does not apply the HR/SpO2 demonstration classifier
    to arbitrary sensor pairs. It returns the relational divergence state directly;
    domain-specific predictive models must be trained and validated on the relevant
    labeled cohort before clinical or outcome claims are made.
    """
    n = len(payload.time_s)
    if n < 10 or not (len(payload.signal_a) == len(payload.signal_b) == n):
        raise HTTPException(
            422,
            "time_s, signal_a and signal_b must have equal length >= 10",
        )
    try:
        features = build_pair_features(
            pd.DataFrame(
                {
                    "time_s": payload.time_s,
                    "a": payload.signal_a,
                    "b": payload.signal_b,
                }
            ),
            "a",
            "b",
            target_hz=payload.target_hz,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    score = float(features["divergence"].iloc[-1])
    return PairAnalyzeResponse(
        signal_a_name=payload.signal_a_name,
        signal_b_name=payload.signal_b_name,
        samples_analyzed=int(len(features)),
        divergence_score=score,
        divergence_velocity=float(features["divergence_velocity"].iloc[-1]),
        divergence_acceleration=float(features["divergence_acceleration"].iloc[-1]),
        risk_flag=bool(score >= payload.divergence_threshold),
        interpretation=(
            "research divergence threshold exceeded"
            if score >= payload.divergence_threshold
            else "research divergence threshold not exceeded"
        ),
    )
