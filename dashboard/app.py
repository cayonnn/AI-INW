# dashboard/app.py
"""
🏆 Guardian AI Trading Dashboard - Pro Edition
==============================================

Professional-grade trading dashboard with REAL MT5 DATA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sqlite3
from datetime import datetime, timedelta

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Guardian AI | Pro Trading",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DARK THEME CSS
# =============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d44;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e30 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] { color: #8888aa; border-radius: 8px; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .status-live {
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        color: #0f0f1a;
        display: inline-block;
    }
    .status-stopped {
        background: #ff4466;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# REAL DATA LOADING
# =============================================================================

@st.cache_data(ttl=5)
def load_guardian_metrics():
    """Load REAL Guardian metrics from CSV."""
    try:
        df = pd.read_csv("logs/guardian_metrics.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        st.warning(f"Guardian metrics not found: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_decision_trace():
    """Load REAL decision trace from log files."""
    try:
        # Get project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try to read alpha_decisions.log (JSON lines format)
        alpha_log_path = os.path.join(project_root, "logs", "decisions", "alpha_decisions.log")
        
        if os.path.exists(alpha_log_path):
            decisions = []
            with open(alpha_log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            import json
                            data = json.loads(line)
                            decisions.append({
                                'timestamp': data.get('timestamp', ''),
                                'cycle': data.get('cycle', 0),
                                'symbol': data.get('symbol', ''),
                                'alpha_action': data.get('ai_signal', data.get('candidate', '')),
                                'alpha_confidence': data.get('ai_confidence', 0),
                                'rule_signal': data.get('rule_signal', ''),
                                'match': data.get('match', False)
                            })
                        except:
                            pass
            
            if decisions:
                df = pd.DataFrame(decisions)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                return df.sort_values('timestamp', ascending=False)
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=10)
def load_trades_db():
    """Load REAL trades from MT5 history."""
    
    # Symbol filter - only show trades for these symbols
    ALLOWED_SYMBOLS = ["XAUUSD", "XAUUSD."]  # Gold only
    
    # Try MT5 first
    if HAS_MT5:
        try:
            if not mt5.initialize():
                st.warning("MT5 not initialized. Using shadow trades.")
            else:
                # Get trades from last 30 days
                from_date = datetime.now() - timedelta(days=30)
                deals = mt5.history_deals_get(from_date, datetime.now())
                
                if deals and len(deals) > 0:
                    trades_list = []
                    for d in deals:
                        # Filter by allowed symbols (XAUUSD)
                        if d.symbol and any(sym in d.symbol for sym in ALLOWED_SYMBOLS):
                            trades_list.append({
                                "ticket": d.ticket,
                                "timestamp": datetime.fromtimestamp(d.time),
                                "symbol": d.symbol,
                                "action": "BUY" if d.type == 0 else "SELL",
                                "lot_size": d.volume,
                                "entry_price": d.price,
                                "pnl": d.profit,
                                "status": "OPEN" if d.profit == 0 else "CLOSED"
                            })
                    
                    if trades_list:
                        df = pd.DataFrame(trades_list)
                        df = df.sort_values('timestamp', ascending=False)
                        return df
        except Exception as e:
            st.warning(f"MT5 Error: {e}")
    
    # Fallback to shadow trades
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "logs", "shadow_trades.csv")
        
        df = pd.read_csv(csv_path, index_col=False)
        if 'open_time' in df.columns:
            df = df.rename(columns={
                'open_time': 'timestamp',
                'type': 'action',
                'volume': 'lot_size',
                'open_price': 'entry_price',
                'close_price': 'exit_price'
            })
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df.sort_values('timestamp', ascending=False)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_shadow_trades():
    """Load REAL shadow trades."""
    try:
        df = pd.read_csv("logs/shadow_trades.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_equity_history():
    """Load REAL equity history."""
    try:
        df = pd.read_csv("logs/equity_history.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def get_system_status():
    """Get REAL system status."""
    status = {
        'alpha_ppo': os.path.exists("models/alpha_ppo_v1.zip"),
        'guardian_ppo': os.path.exists("models/guardian_ppo_v3_20260115_1857.zip"),
        'xgb': os.path.exists("models/xgb_imitation.pkl"),
        'is_live': os.path.exists("heartbeat.txt"),
    }
    
    # Get last training time
    try:
        meta_files = [f for f in os.listdir("runs/alpha_v1") if os.path.isdir(f"runs/alpha_v1/{f}")]
        if meta_files:
            latest = max(meta_files, key=lambda x: os.path.getmtime(f"runs/alpha_v1/{x}"))
            status['last_train'] = latest.replace("alpha_ppo_v1_", "").replace("_", " ")
        else:
            status['last_train'] = "N/A"
    except:
        status['last_train'] = "N/A"
    
    return status

# =============================================================================
# LOAD ALL DATA
# =============================================================================

metrics_df = load_guardian_metrics()
decision_df = load_decision_trace()
trades_df = load_trades_db()
shadow_df = load_shadow_trades()
equity_df = load_equity_history()
status = get_system_status()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 2.5rem; margin: 0;">🛡️</h1>
        <h2 style="font-size: 1.5rem; margin: 10px 0;">Guardian AI</h2>
        <p style="color: #8888aa;">Pro Trading System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if status['is_live']:
        st.markdown('<div class="status-live">🟢 LIVE TRADING</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-stopped">🔴 STOPPED</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🧠 Models")
    st.markdown(f"Alpha PPO: {'✅' if status['alpha_ppo'] else '❌'}")
    st.markdown(f"Guardian: {'✅' if status['guardian_ppo'] else '❌'}")
    st.markdown(f"XGBoost: {'✅' if status['xgb'] else '❌'}")
    
    st.markdown("---")
    
    st.markdown("### 📅 Training")
    st.caption(f"Last: {status['last_train']}")
    
    st.markdown("---")
    
    st.markdown("### 📂 Data Status")
    st.caption(f"Guardian Logs: {len(metrics_df)} rows")
    st.caption(f"Decisions: {len(decision_df)} rows")
    st.caption(f"Trades: {len(trades_df)} rows")
    st.caption(f"Shadow: {len(shadow_df)} rows")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="font-size: 2.5rem; margin: 0;">🛡️ Guardian AI Trading</h1>
    <p style="color: #8888aa; font-size: 1rem;">Professional Trading Dashboard | Real-Time MT5</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# GET MT5 ACCOUNT INFO
# =============================================================================

mt5_balance = 0
mt5_equity = 0
mt5_profit = 0
mt5_margin_level = 0
mt5_drawdown = 0

if HAS_MT5 and mt5.initialize():
    account = mt5.account_info()
    if account:
        mt5_balance = account.balance
        mt5_equity = account.equity
        mt5_profit = account.profit
        mt5_margin_level = account.margin_level if account.margin_level else 0
        # Calculate drawdown
        if account.balance > 0:
            mt5_drawdown = ((account.balance - account.equity) / account.balance * 100)
            mt5_drawdown = max(0, mt5_drawdown)

# =============================================================================
# CALCULATE TRADING METRICS
# =============================================================================

total_pnl = 0
total_trades = 0
wins = 0
losses = 0
win_rate = 0
avg_win = 0
avg_loss = 0
profit_factor = 0
max_profit = 0
max_loss = 0
max_drawdown = 0
expectancy = 0

if not trades_df.empty and 'pnl' in trades_df.columns:
    total_pnl = trades_df['pnl'].sum()
    total_trades = len(trades_df)
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    wins = len(winning_trades)
    losses = len(losing_trades)
    win_rate = wins / max(total_trades, 1) * 100
    
    # Average win/loss
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    # Profit Factor
    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Max profit/loss
    max_profit = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    # Calculate max drawdown from equity curve
    if 'timestamp' in trades_df.columns:
        curve_df = trades_df.sort_values('timestamp', ascending=True).copy()
        curve_df['cumulative_pnl'] = curve_df['pnl'].cumsum()
        curve_df['peak'] = curve_df['cumulative_pnl'].cummax()
        curve_df['drawdown'] = curve_df['peak'] - curve_df['cumulative_pnl']
        max_drawdown = curve_df['drawdown'].max()

# Handle NaN values
total_pnl = 0 if pd.isna(total_pnl) else total_pnl
avg_win = 0 if pd.isna(avg_win) else avg_win
avg_loss = 0 if pd.isna(avg_loss) else avg_loss
max_profit = 0 if pd.isna(max_profit) else max_profit
max_loss = 0 if pd.isna(max_loss) else max_loss
expectancy = 0 if pd.isna(expectancy) else expectancy

# =============================================================================
# TOP ROW: MT5 ACCOUNT METRICS
# =============================================================================

st.markdown("### 💼 MT5 Account")
col_a1, col_a2, col_a3, col_a4 = st.columns(4)

with col_a1:
    st.metric("💰 Balance", f"${mt5_balance:,.2f}")
with col_a2:
    delta_color = "normal" if mt5_equity >= mt5_balance else "inverse"
    st.metric("📊 Equity", f"${mt5_equity:,.2f}", delta=f"${mt5_equity - mt5_balance:,.2f}")
with col_a3:
    st.metric("💹 Floating P&L", f"${mt5_profit:,.2f}")
with col_a4:
    st.metric("📉 Drawdown", f"{mt5_drawdown:.2f}%")

st.markdown("---")

# =============================================================================
# SECOND ROW: TRADING PERFORMANCE
# =============================================================================

st.markdown("### 📈 Trading Performance")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    color = "🟢" if total_pnl >= 0 else "🔴"
    st.metric(f"{color} Total P&L", f"${total_pnl:,.2f}")
with col2:
    st.metric("📊 Total Trades", f"{total_trades}")
with col3:
    wr_color = "🟢" if win_rate >= 50 else "🔴"
    st.metric(f"{wr_color} Win Rate", f"{win_rate:.1f}%", delta=f"{wins}W / {losses}L")
with col4:
    pf_color = "🟢" if profit_factor >= 1.5 else "🟡" if profit_factor >= 1 else "🔴"
    st.metric(f"{pf_color} Profit Factor", f"{profit_factor:.2f}")
with col5:
    st.metric("📈 Max DD", f"${max_drawdown:,.2f}")
with col6:
    exp_color = "🟢" if expectancy >= 0 else "🔴"
    st.metric(f"{exp_color} Expectancy", f"${expectancy:,.2f}")

st.markdown("---")

# =============================================================================
# THIRD ROW: DETAILED STATS
# =============================================================================

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric("🏆 Best Trade", f"${max_profit:,.2f}")
with col_s2:
    st.metric("💔 Worst Trade", f"${max_loss:,.2f}")
with col_s3:
    st.metric("📈 Avg Win", f"${avg_win:,.2f}")
with col_s4:
    st.metric("📉 Avg Loss", f"${avg_loss:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# CHARTS (REAL DATA)
# =============================================================================

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📈 Equity Curve (Real MT5)")
    
    # Try to get current MT5 equity
    current_balance = 0
    current_equity = 0
    if HAS_MT5 and mt5.initialize():
        account = mt5.account_info()
        if account:
            current_balance = account.balance
            current_equity = account.equity
    
    # Calculate equity curve from trades
    if not trades_df.empty and 'pnl' in trades_df.columns and 'timestamp' in trades_df.columns:
        # Sort by timestamp ascending
        curve_df = trades_df.sort_values('timestamp', ascending=True).copy()
        
        # Calculate starting balance (current balance minus cumulative PnL)
        total_pnl = curve_df['pnl'].sum()
        starting_balance = current_balance - total_pnl if current_balance > 0 else 1000
        
        # Calculate cumulative equity
        curve_df['cumulative_pnl'] = curve_df['pnl'].cumsum()
        curve_df['equity'] = starting_balance + curve_df['cumulative_pnl']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df['timestamp'],
            y=curve_df['equity'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#00d4ff', width=2),
            fillcolor='rgba(0, 212, 255, 0.1)',
            name='Equity'
        ))
        
        # Add current equity point
        if current_equity > 0:
            fig.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[current_equity],
                mode='markers',
                marker=dict(size=12, color='#00ff88', symbol='star'),
                name=f'Current: ${current_equity:.2f}'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#2d2d44', title='Equity ($)'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    elif current_equity > 0:
        # Just show current equity as single point
        st.metric("Current Equity", f"${current_equity:,.2f}")
        st.metric("Current Balance", f"${current_balance:,.2f}")
    else:
        st.info("No equity data available yet")

with col_right:
    st.markdown("### 📊 Daily P&L (Last 7 Days)")
    
    if not trades_df.empty and 'pnl' in trades_df.columns and 'timestamp' in trades_df.columns:
        # Use timestamp (open_time) for grouping - close_time has parsing issues
        date_col = 'timestamp'
        
        # Create a copy to avoid modifying original
        chart_df = trades_df.copy()
        
        # Ensure the date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(chart_df[date_col]):
            chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors='coerce')
        
        # Filter to last 7 days
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        chart_df = chart_df[chart_df[date_col] >= week_ago]
        
        # Extract date only
        chart_df['date'] = chart_df[date_col].dt.date
        
        # Group by date and sum PnL
        daily_pnl_df = chart_df.groupby('date')['pnl'].sum().reset_index()
        daily_pnl_df = daily_pnl_df.dropna()
        daily_pnl_df = daily_pnl_df.sort_values('date')
        
        if not daily_pnl_df.empty:
            colors = ['#00ff88' if x >= 0 else '#ff4466' for x in daily_pnl_df['pnl']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_pnl_df['date'],
                y=daily_pnl_df['pnl'],
                marker_color=colors,
                text=[f"${x:.2f}" for x in daily_pnl_df['pnl']],
                textposition='outside'
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=300,
                yaxis=dict(title="Profit ($)"),
                xaxis=dict(
                    tickformat='%d %b',
                    dtick='D1'  # Show each day
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades in the last 7 days")
    else:
        st.info("No trade data available yet")

# =============================================================================
# TABS (REAL DATA)
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Trades", "🛡️ Guardian", "🧠 Alpha PPO", "⚙️ System"
])

with tab1:
    st.markdown("### Recent Trades (Real)")
    
    if not trades_df.empty:
        st.dataframe(trades_df.head(20), use_container_width=True, height=400)
    else:
        st.info("No trades recorded yet")
    
    # Trade Statistics
    if not trades_df.empty and 'pnl' in trades_df.columns:
        st.markdown("### Trade Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades_df))
        with col2:
            st.metric("Total P&L", f"${trades_df['pnl'].sum():.2f}")
        with col3:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            st.metric("Avg Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "$0")
        with col4:
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            st.metric("Avg Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "$0")

with tab2:
    st.markdown("### Guardian Decision Log (Real)")
    
    if not metrics_df.empty:
        st.dataframe(metrics_df.tail(30), use_container_width=True, height=300)
        
        # Guardian Stats
        st.markdown("### Guardian Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'action' in metrics_df.columns:
                blocks = len(metrics_df[metrics_df['action'] == 'BLOCK'])
                st.metric("Total Blocks", blocks)
            else:
                st.metric("Total Logs", len(metrics_df))
        with col2:
            if 'dd_pct' in metrics_df.columns:
                max_dd = metrics_df['dd_pct'].max()
                st.metric("Max DD", f"{max_dd:.2f}%")
            elif 'daily_dd' in metrics_df.columns:
                max_dd = metrics_df['daily_dd'].max()
                st.metric("Max DD", f"{max_dd:.2f}%")
        with col3:
            if 'equity' in metrics_df.columns:
                current_eq = metrics_df['equity'].iloc[-1]
                st.metric("Current Equity", f"${current_eq:,.2f}")
    else:
        st.info("No Guardian logs yet")
    
    # Decision Trace
    if not decision_df.empty:
        st.markdown("### Alpha-Guardian Decision Trace")
        st.dataframe(decision_df.tail(20), use_container_width=True)

with tab3:
    st.markdown("### Alpha PPO Decision Log (Real)")
    
    if not decision_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Decisions", len(decision_df))
            
            if 'alpha_confidence' in decision_df.columns:
                avg_conf = decision_df['alpha_confidence'].astype(float).mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
        
        with col2:
            if 'alpha_action' in decision_df.columns:
                action_dist = decision_df['alpha_action'].value_counts()
                st.markdown("**Action Distribution:**")
                for action, count in action_dist.items():
                    st.markdown(f"- {action}: {count}")
        
        # Confidence Chart
        if 'alpha_confidence' in decision_df.columns and 'timestamp' in decision_df.columns:
            st.markdown("### Confidence Over Time")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=decision_df['timestamp'],
                y=decision_df['alpha_confidence'].astype(float),
                mode='lines+markers',
                line=dict(color='#7c3aed', width=2),
                marker=dict(size=4)
            ))
            fig.add_hline(y=0.55, line_dash="dash", line_color="#ff4466",
                         annotation_text="Threshold")
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Alpha PPO decisions logged yet")
    
    # Shadow Trades
    if not shadow_df.empty:
        st.markdown("### Shadow Trades")
        st.dataframe(shadow_df.tail(10), use_container_width=True)

with tab4:
    st.markdown("### System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trading Config**")
        try:
            import yaml
            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, "config", "live_alpha_ppo.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            st.json(config.get("risk", {}))
        except Exception as e:
            st.code(f"Config file not found: {e}")
    
    with col2:
        st.markdown("**Model Files**")
        models = []
        try:
            for f in os.listdir("models"):
                if f.endswith(".zip") or f.endswith(".pkl"):
                    size = os.path.getsize(f"models/{f}") / 1024 / 1024
                    models.append({"file": f, "size_mb": f"{size:.2f}"})
            st.dataframe(pd.DataFrame(models), use_container_width=True)
        except:
            st.warning("Cannot list models")
    
    st.markdown("### Log Files")
    
    logs = []
    try:
        for f in os.listdir("logs"):
            if f.endswith(".csv"):
                size = os.path.getsize(f"logs/{f}") / 1024
                logs.append({"file": f, "size_kb": f"{size:.1f}"})
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    except:
        st.warning("Cannot list logs")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.markdown(f"""
<div style="text-align: center; padding: 20px; color: #8888aa;">
    <p>🛡️ Guardian AI Trading System v3.0 Pro | Real Data Mode</p>
    <p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)
