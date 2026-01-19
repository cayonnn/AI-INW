# dashboard_v4.py
"""
Guardian V4 Dashboard - Multi-Symbol Monitor
============================================

Real-time visualization for live_loop_v4.py.
Displays a 4-symbol grid with consolidated risk metrics.

Run: python -m streamlit run dashboard_v4.py
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

# =========================================
# Configuration
# =========================================
st.set_page_config(
    page_title="🛡️ Guardian V4 Monitor",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Theme & Custom CSS (Commented out for debug, re-enabling safe parts)
st.markdown("""
<style>
    /* Metrics */
    .metric-card {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }
    
    /* Status Colors */
    .status-ALLOW { border-left: 5px solid #00ff00; }
    .status-BLOCK { border-left: 5px solid #ff4b4b; }
    .status-FORCE_HOLD { border-left: 5px solid #ffa500; }
    .status-EMERGENCY_FREEZE { border-left: 5px solid #ff0000; }
    
    /* Symbol Card */
    .symbol-card {
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #444;
        height: 100%;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# Data Loader
# =========================================
@st.cache_data(ttl=2)
def load_data():
    path = Path("logs/guardian_multisymbol.csv")
    columns = [
        "timestamp", "cycle", "symbol", "action", "confidence", 
        "equity", "margin_ratio", "block_reason", "daily_dd"
    ]
    
    if not path.exists():
        return pd.DataFrame(columns=columns)
    
    try:
        # Try reading with header
        df = pd.read_csv(path)
        
        # Check if expected columns exist
        if "cycle" not in df.columns:
            # Fallback: Read without header
            df = pd.read_csv(path, header=None, names=columns)
            
        return df
    except Exception as e:
        # Return empty DF on error to avoid crash
        return pd.DataFrame()

# =========================================
# Main Layout
# =========================================
st.title("🛡️ Guardian V4 | Multi-Symbol Engine")

# Controls
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

# Load Data
df = load_data()

# Debug: Show raw data status (Collapsed by default)
with st.expander("🛠️ Debug Data View", expanded=False):
    if not df.empty:
        st.write(f"Rows: {len(df)}")
        st.write(df.head())
    else:
        st.write("No data loaded.")

if df.empty:
    st.warning("⏳ Waiting for Live Loop V4 data... (logs/guardian_multisymbol.csv)")
    if auto_refresh:
        time.sleep(2)
        st.rerun()
    st.stop()

# Clean Data Types
try:
    # Ensure cycle is int
    df['cycle'] = pd.to_numeric(df['cycle'], errors='coerce')
    df = df.dropna(subset=['cycle'])
    df['cycle'] = df['cycle'].astype(int)
except Exception as e:
    st.error(f"Data Cleaning Error: {e}")
    st.stop()

# 1. Global Health Header
# -----------------------
latest_cycle = df['cycle'].max()
current_snapshot = df[df['cycle'] == latest_cycle] # Snapshot of last cycle across symbols
global_equity = current_snapshot['equity'].max() if not current_snapshot.empty else 0
global_dd = current_snapshot['daily_dd'].max() if not current_snapshot.empty else 0
margin_level = current_snapshot['margin_ratio'].min() * 100 if not current_snapshot.empty else 0

cols = st.columns(4)
with cols[0]:
    st.metric("💵 Total Equity", f"${global_equity:,.2f}")
with cols[1]:
    st.metric("📉 Global DD", f"{global_dd:.2f}%", delta="-Limits", delta_color="inverse")
with cols[2]:
    st.metric("💧 Free Margin", f"{margin_level:.1f}%")
with cols[3]:
    st.metric("🔄 Cycle", f"#{latest_cycle}")

st.markdown("---")

# 2. Symbol Grid (2x2)
# --------------------
SYMBOLS = ["XAUUSD", "EURUSD", "BTCUSD", "GBPJPY"]

# Row 1
r1c1, r1c2 = st.columns(2)
# Row 2
r2c1, r2c2 = st.columns(2)

grid_map = {
    "XAUUSD": r1c1,
    "EURUSD": r1c2,
    "BTCUSD": r2c1,
    "GBPJPY": r2c2
}

for sym in SYMBOLS:
    with grid_map.get(sym, st.container()):
        # Get latest data for symbol
        sym_data = df[df['symbol'] == sym]
        
        if not sym_data.empty:
            last = sym_data.iloc[-1]
            status = last['action']
            reason = last['block_reason']
            # Safely get confidence
            try:
                conf = float(last.get('confidence', 0.0))
            except:
                conf = 0.0
            
            # Color coding
            status_color = "green" if status == "ALLOW" else "red" if "BLOCK" in status else "orange"
            border_cls = f"status-{status}"
            
            st.markdown(f"""
            <div class='symbol-card {border_cls}'>
                <h3 style='margin:0; color:white;'>{sym}</h3>
                <p style='color:{status_color}; font-weight:bold; font-size:18px;'>{status}</p>
                <p style='color:#888; font-size:12px;'>Reason: {reason if reason else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mini Charts (Confidence/Activity)
            st.caption("Recent Activity (Action)")
            try:
                chart_data = sym_data.tail(50).copy()
                chart_data['allow_flag'] = chart_data['action'].apply(lambda x: 1 if x == "ALLOW" else 0)
                st.area_chart(chart_data['allow_flag'], height=50)
            except:
                st.write("Chart Error")

        else:
            st.markdown(f"""
            <div class='symbol-card'>
                <h3 style='color:grey;'>{sym}</h3>
                <p>No Data</p>
            </div>
            """, unsafe_allow_html=True)

# 3. Recent Log Table (Sidebar or Bottom)
# ---------------------------------------
with st.expander("📝 System Logs (Last 10 Events)", expanded=False):
    st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)

# =========================================
# Auto Refresh Logic (Last Step)
# =========================================
if auto_refresh:
    time.sleep(2)
    st.rerun()
