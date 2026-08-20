import plotly.graph_objects as go
import streamlit as st

from diverge.data import generate_synthetic_record
from diverge.detection import predict_detector, threshold_baseline, train_detector
from diverge.features import build_features

st.set_page_config(page_title="DiVerge", layout="wide")
st.title("DiVerge — Inter-Sensor Physiological Divergence")
st.caption(
    "Sensor-agnostic research framework. The dashboard uses a synthetic HR/SpO2 pair only as an "
    "illustrative engineering example; it is not a medical device or clinical decision system."
)

st.info(
    "Core hypothesis: risk may first appear as a change in the relationship between normally coupled "
    "signals, before either channel individually crosses an abnormal threshold."
)

seed = st.sidebar.number_input("Synthetic demonstration seed", 0, 10000, 42)
rec = generate_synthetic_record(seed=int(seed))
frame = build_features(rec.frame)
model = train_detector(build_features(generate_synthetic_record(seed=7).frame))
div = predict_detector(model, frame)
base = threshold_baseline(frame)
frame["risk_confidence"] = div.confidence

left, right = st.columns(2)
with left:
    st.subheader("Illustrative physiological pair")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.time_s, y=frame.hr, name="Heart rate (example signal A)"))
    fig.add_trace(go.Scatter(x=frame.time_s, y=frame.spo2, name="SpO2 (example signal B)"))
    fig.add_vline(x=rec.event_start, line_dash="dash", annotation_text="Injected event")
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Raw signal value")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Relational state")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=frame.time_s, y=frame.divergence, name="Fused divergence"))
    fig2.add_trace(go.Scatter(x=frame.time_s, y=frame.risk_confidence, name="Demo risk confidence"))
    fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Research score")
    st.plotly_chart(fig2, use_container_width=True)

div_t = frame.loc[div.risk, "time_s"].min() if div.risk.any() else None
base_t = frame.loc[base.risk, "time_s"].min() if base.risk.any() else None
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Divergence-model alert", f"{div_t:.0f}s" if div_t is not None else "none")
with m2:
    st.metric("Single-signal threshold alert", f"{base_t:.0f}s" if base_t is not None else "none")
with m3:
    if div_t is not None and base_t is not None:
        st.metric("Illustrative lead-time difference", f"{base_t - div_t:.0f}s")
    else:
        st.metric("Illustrative lead-time difference", "n/a")

st.markdown(
    """
### How to interpret this demo

The synthetic generator intentionally creates relational breakdown before overt threshold abnormalities. Its
purpose is to validate the software path, not to prove clinical effectiveness. Real scientific evidence must
come from prespecified labeled datasets and held-out evaluation. CTU-CHB is implemented as one such
validation track; future datasets can use other physiological signal pairs without changing the core idea.
"""
)
