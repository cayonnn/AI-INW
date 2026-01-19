import streamlit as st
import numpy as np
import plotly.express as px

FEATURES = [
    "Equity",
    "Free Margin",
    "Daily DD",
    "Error Rate",
    "Latency"
]

def render_attention_map(latest_row):
    # Check if attention columns exist
    if not all(hasattr(latest_row, f"att_{k}") for k in ["equity", "margin", "dd", "error", "latency"]):
        st.info("⚠️ PPO Attention data not available yet (waiting for next PPO inference)")
        return

    # Mock attention weights if not strictly logged, or use logged values
    # Using getattr with default 0.2 to avoid crash if some missing
    weights = np.array([
        getattr(latest_row, "att_equity", 0.2),
        getattr(latest_row, "att_margin", 0.2),
        getattr(latest_row, "att_dd", 0.2),
        getattr(latest_row, "att_error", 0.2),
        getattr(latest_row, "att_latency", 0.2),
    ])

    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(5) / 5

    fig = px.bar(
        x=FEATURES,
        y=weights,
        title="PPO Attention Weights (Decision Importance)",
        labels={"x": "State Feature", "y": "Attention Weight"}
    )
    
    # Competition styling
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**
    - Higher bar = PPO considered this signal more important
    - Emergency freezes usually correlate with Margin + DD dominance
    """)
