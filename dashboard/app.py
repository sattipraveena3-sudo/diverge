import plotly.graph_objects as go
import streamlit as st

from diverge.data import generate_synthetic_record
from diverge.detection import predict_detector, threshold_baseline, train_detector
from diverge.features import build_features

st.set_page_config(page_title="DiVerge", layout="wide")
st.title("DiVerge — Physiological Relational Divergence")
st.caption("Research demo only. Not a medical device and not for clinical decision-making.")
seed = st.sidebar.number_input("Synthetic seed", 0, 10000, 42)
rec = generate_synthetic_record(seed=int(seed))
frame = build_features(rec.frame)
model = train_detector(build_features(generate_synthetic_record(seed=7).frame))
div = predict_detector(model, frame)
base = threshold_baseline(frame)
frame["risk_confidence"] = div.confidence
fig = go.Figure()
fig.add_trace(go.Scatter(x=frame.time_s, y=frame.hr, name="Heart rate"))
fig.add_trace(go.Scatter(x=frame.time_s, y=frame.spo2, name="SpO2"))
fig.add_vline(x=rec.event_start, line_dash="dash", annotation_text="Adverse event")
st.plotly_chart(fig, use_container_width=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=frame.time_s, y=frame.divergence, name="Divergence"))
fig2.add_trace(go.Scatter(x=frame.time_s, y=frame.risk_confidence, name="Risk confidence"))
st.plotly_chart(fig2, use_container_width=True)
div_t = frame.loc[div.risk, "time_s"].min() if div.risk.any() else None
base_t = frame.loc[base.risk, "time_s"].min() if base.risk.any() else None
st.metric("Divergence alert", f"{div_t:.0f}s" if div_t is not None else "none")
st.metric("Threshold baseline alert", f"{base_t:.0f}s" if base_t is not None else "none")
