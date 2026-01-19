# dashboard_guardian.py
"""
Guardian Dashboard - Streamlit
================================

Realtime Guardian vs Alpha visualization.
Competition-ready dashboard for presentation.

Run: streamlit run dashboard_guardian.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="🛡️ Guardian Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #00d9ff !important;
    }
    .main {
        background-color: #0f0f23;
    }
    h1, h2, h3 {
        color: #00d9ff;
    }
    .status-allow { color: #4ecca3; font-weight: bold; }
    .status-hold { color: #ffd93d; font-weight: bold; }
    .status-kill { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛡️ Guardian vs Alpha Dashboard")
st.markdown("**Autonomous Trading Governance System**")

# Try to load metrics
metrics_path = Path("logs/guardian_metrics.csv")

if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    
    if len(df) > 0:
        latest = df.iloc[-1]
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dd_val = latest.get('daily_dd', 0) * 100
            st.metric("📉 Daily DD", f"{dd_val:.2f}%", 
                     delta=f"{dd_val - df['daily_dd'].iloc[-2]*100:.2f}%" if len(df) > 1 else None,
                     delta_color="inverse")
        
        with col2:
            margin_val = latest.get('margin_ratio', 0) * 100
            st.metric("💰 Margin", f"{margin_val:.1f}%")
        
        with col3:
            equity = latest.get('equity', 0)
            st.metric("💵 Equity", f"${equity:.2f}")
        
        with col4:
            action = latest.get('action', 'ALLOW')
            status_class = "status-allow" if action == "ALLOW" else "status-hold" if "HOLD" in str(action) else "status-kill"
            st.metric("🛡️ Guardian Status", action)
        
        st.markdown("---")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📈 Risk Timeline")
            chart_df = df[['daily_dd', 'margin_ratio']].copy()
            chart_df.columns = ['DD', 'Margin']
            st.line_chart(chart_df)
        
        with col_right:
            st.subheader("🛡️ Guardian Actions")
            if 'action' in df.columns:
                action_counts = df['action'].value_counts()
                st.bar_chart(action_counts)
        
        st.markdown("---")
        
        # Event log
        st.subheader("📋 Recent Events")
        recent = df.tail(10)[['timestamp', 'cycle', 'action', 'block_reason', 'escalation']].copy()
        recent = recent.iloc[::-1]  # Reverse order
        st.dataframe(recent, use_container_width=True)
        
        # Stats
        st.markdown("---")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            total_blocks = (df['block_count'] > 0).sum() if 'block_count' in df.columns else 0
            st.metric("🚫 Total Blocks", total_blocks)
        
        with col_s2:
            escalations = df['escalation'].sum() if 'escalation' in df.columns else 0
            st.metric("⬆️ Escalations", int(escalations))
        
        with col_s3:
            cycles = len(df)
            st.metric("🔄 Cycles", cycles)
        
        with col_s4:
            kill_count = (df['action'] == 'KILL_SWITCH').sum() if 'action' in df.columns else 0
            st.metric("☠️ Kill Switch", kill_count)
    
    else:
        st.warning("No data in metrics file yet. Run live_loop_v3.py to generate data.")

else:
    st.error("Metrics file not found: logs/guardian_metrics.csv")
    st.info("Run `python live_loop_v3.py --live` to generate Guardian metrics.")
    
    # Demo mode
    st.markdown("---")
    st.subheader("📊 Demo Mode")
    
    # Generate fake data for demo
    demo_data = {
        'cycle': range(1, 21),
        'daily_dd': np.cumsum(np.random.uniform(0, 0.02, 20)),
        'margin_ratio': 0.8 - np.cumsum(np.random.uniform(0, 0.02, 20)),
        'action': np.random.choice(['ALLOW', 'ALLOW', 'REDUCE_RISK', 'FORCE_HOLD'], 20)
    }
    demo_df = pd.DataFrame(demo_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(demo_df[['daily_dd', 'margin_ratio']])
    
    with col2:
        st.bar_chart(demo_df['action'].value_counts())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🛡️ Guardian AI - Autonomous Trading Governance System | Competition 2026"
    "</div>",
    unsafe_allow_html=True
)
