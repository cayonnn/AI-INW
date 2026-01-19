# dashboard_advanced.py
"""
Guardian Advanced Dashboard - Competition Grade
=================================================

Features:
- KPI Layer (Rule vs PPO)
- Live time-series
- Attention map
- DD efficiency
- Export figures

Run: streamlit run dashboard_advanced.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# =============================================================================
# Config
# =============================================================================

st.set_page_config(
    page_title="🛡️ Guardian Advanced Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 10px; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Guardian Control Dashboard")
st.markdown("**Rule vs PPO Hybrid Analysis**")

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# File upload or demo
data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Demo Data"])

if data_source == "Upload CSV":
    file = st.sidebar.file_uploader("Upload guardian_metrics.csv", type=["csv"])
    if file is None:
        st.info("📂 Upload guardian metrics CSV to begin")
        st.stop()
    df = load_csv(file)
else:
    # Generate demo data
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-15 09:00', periods=n, freq='1min'),
        'dd': np.cumsum(np.random.uniform(-0.001, 0.003, n)).clip(0, 0.15),
        'margin': (0.9 - np.cumsum(np.random.uniform(0, 0.005, n))).clip(0.2, 1),
        'equity': 1000 + np.cumsum(np.random.uniform(-5, 10, n)),
        'balance': 1000 + np.cumsum(np.random.uniform(0, 3, n)),
        'rule_action': np.random.choice(['ALLOW', 'ALLOW', 'ALLOW', 'REDUCE_RISK', 'FORCE_HOLD'], n),
        'ppo_action': np.random.choice(['ALLOW', 'ALLOW', 'REDUCE_RISK', 'FORCE_HOLD'], n),
        'ppo_conf': np.random.uniform(0.4, 0.95, n),
        'freeze': np.random.choice([False, False, False, False, True], n),
    })
    st.sidebar.success("Using demo data")

# =============================================================================
# KPI Metrics
# =============================================================================

st.markdown("---")

total = len(df)
rule_blocks = (df['rule_action'] != 'ALLOW').sum() if 'rule_action' in df.columns else 0
ppo_blocks = (df['ppo_action'] != 'ALLOW').sum() if 'ppo_action' in df.columns else 0
freeze_time = df['freeze'].sum() if 'freeze' in df.columns else 0
max_dd = df['dd'].max() * 100 if 'dd' in df.columns else 0

# PPO early intervention
ppo_early = 0
if 'ppo_action' in df.columns and 'rule_action' in df.columns:
    ppo_early = ((df['ppo_action'] != 'ALLOW') & (df['rule_action'] == 'ALLOW')).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🛡️ Rule Blocks", rule_blocks)
c2.metric("🧠 PPO Blocks", ppo_blocks)
c3.metric("⏸️ Freeze Time", f"{freeze_time}s")
c4.metric("📉 Max DD", f"{max_dd:.2f}%")
c5.metric("⚡ PPO Early", ppo_early)

# =============================================================================
# Time Series Charts
# =============================================================================

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Equity & Balance")
    if 'equity' in df.columns:
        fig = px.line(df, x='timestamp', y=['equity', 'balance'] if 'balance' in df.columns else ['equity'])
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📉 Drawdown")
    if 'dd' in df.columns:
        fig = px.area(df, x='timestamp', y='dd', title=None)
        fig.update_traces(fill='tozeroy', line_color='red')
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Action Comparison
# =============================================================================

st.markdown("---")
st.subheader("🧠 Rule vs PPO Actions")

action_map = {'ALLOW': 0, 'REDUCE_RISK': 1, 'FORCE_HOLD': 2, 'EMERGENCY_FREEZE': 3}

if 'rule_action' in df.columns and 'ppo_action' in df.columns:
    df['rule_code'] = df['rule_action'].map(action_map).fillna(0)
    df['ppo_code'] = df['ppo_action'].map(action_map).fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rule_code'], mode='lines', name='Rule', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ppo_code'], mode='lines', name='PPO', line=dict(color='green', dash='dot')))
    fig.update_layout(yaxis_title='Action Level', xaxis_title='Time')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PPO Confidence Distribution
# =============================================================================

col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 PPO Confidence")
    if 'ppo_conf' in df.columns:
        fig = px.histogram(df, x='ppo_conf', nbins=30)
        fig.add_vline(x=0.65, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("🎯 Action Distribution")
    if 'ppo_action' in df.columns:
        counts = df['ppo_action'].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title=None)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# DD Efficiency
# =============================================================================

st.markdown("---")
st.subheader("📉 Risk Reduction Efficiency")

# Calculate efficiency metrics
if 'dd' in df.columns and 'freeze' in df.columns:
    baseline_dd = 0.15  # Hypothetical baseline without guardian
    actual_max_dd = df['dd'].max()
    dd_avoided = baseline_dd - actual_max_dd
    freeze_cost = df['freeze'].sum() / len(df)
    efficiency = dd_avoided / max(freeze_cost, 0.001)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("DD Avoided", f"{dd_avoided*100:.2f}%")
    c2.metric("Freeze Cost", f"{freeze_cost*100:.1f}%")
    c3.metric("Efficiency Ratio", f"{efficiency:.2f}")

# =============================================================================
# Export
# =============================================================================

st.markdown("---")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.download_button(
        "⬇️ Export CSV",
        df.to_csv(index=False),
        "guardian_export.csv",
        "text/csv"
    )

with col_exp2:
    if st.button("🏆 Generate Competition Figures"):
        import matplotlib.pyplot as plt
        
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        
        # DD curve
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(range(len(df)), df['dd'] * 100, alpha=0.3, color='red')
        ax.plot(df['dd'] * 100, color='red')
        ax.set_ylabel('Drawdown %')
        ax.set_xlabel('Time')
        ax.set_title('Guardian Protected Drawdown')
        plt.tight_layout()
        plt.savefig(figures_dir / 'dd_curve.png', dpi=150)
        plt.close()
        
        # Action rate
        if 'ppo_action' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['ppo_action'].value_counts().plot(kind='bar', ax=ax, color=['green', 'orange', 'red', 'darkred'][:len(df['ppo_action'].unique())])
            ax.set_title('Guardian Action Distribution')
            ax.set_ylabel('Count')
            plt.tight_layout()
            plt.savefig(figures_dir / 'action_rate.png', dpi=150)
            plt.close()
        
        st.success(f"✅ Figures saved to {figures_dir.absolute()}")

# =============================================================================
# Raw Data
# =============================================================================

with st.expander("📄 Raw Data (last 100)"):
    st.dataframe(df.tail(100))

# Footer
st.markdown("---")
st.caption("🛡️ Guardian Control Dashboard | Competition 2026")
