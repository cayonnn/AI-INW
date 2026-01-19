# src/dashboard/dashboard_server.py
"""
🏦 HEDGE FUND GRADE Trading Dashboard (Professional V2)
=========================================================
Features:
- Live Financial Metrics & Charts
- Real-time Workflow Visualization (Pipeline State)
- Project Development Tracker (Gantt/Status)
- Log Inspection
"""

import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from collections import deque
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# --- Backend Modules ---
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except:
    HAS_MT5 = False
    mt5 = None

from src.dashboard.project_tracker import ProjectTracker
from src.dashboard.workflow_state import get_workflow_state
from src.utils.analytics import AnalyticsDashboard
from pydantic import BaseModel

# --- Initialization ---
project_tracker = ProjectTracker()
workflow_state = get_workflow_state()
analytics = AnalyticsDashboard()



# --- MT5 Helpers (Reused) ---
def init_mt5():
    if not HAS_MT5: return False
    try: return mt5.initialize()
    except: return False

def get_mt5_account():
    if not HAS_MT5 or not init_mt5(): return {}
    try:
        i = mt5.account_info()
        if not i: return {}
        dd = ((i.balance - i.equity) / i.balance * 100) if i.balance > 0 else 0
        return {
            "balance": round(i.balance, 2), "equity": round(i.equity, 2),
            "margin_level": round(i.margin_level, 2) if i.margin_level else 0,
            "profit": round(i.profit, 2), "server": i.server,
            "drawdown_pct": round(max(0, dd), 2),
        }
    except: return {}

def get_mt5_positions(symbol=None):
    if not HAS_MT5 or not init_mt5(): return []
    try:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions: return []
        result = []
        for p in positions:
            pip = 0.01 if "XAU" in p.symbol else 0.0001
            pips = (p.price_current - p.price_open) / pip
            if p.type == 1: pips = -pips
            result.append({
                "ticket": p.ticket, "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume, "profit": round(p.profit, 2), "pips": round(pips, 1)
            })
        return result
    except: return []

def get_detailed_history(days=30):
    if not HAS_MT5 or not init_mt5(): return {"trades": [], "metrics": {}}
    
    # 1. Fetch from MT5
    from_date = datetime.now() - timedelta(days=days)
    try:
        deals = mt5.history_deals_get(from_date, datetime.now())
        if not deals: return {"trades": [], "metrics": {}}
        
        # 2. Process into Analytics
        # Reset analytics to avoid duplicates in this simple stateless view
        # In a real app we might persist, but here we rebuild from MT5 history
        temp_analytics = AnalyticsDashboard() 
        
        trades_list = []
        
        for d in deals:
            if d.entry == 1: # Entry Out (Close) or Entry In/Out
                # Simplify: just look at profit > 0 or < 0 for realized PnL
                # Note: MT5 deals are individual executions. 
                # Ideally we match entry/exit, but for summary PnL, sum of deals works.
                pass
            
            # Simple conversion for table
            if d.profit != 0:
                t = {
                    "ticket": d.ticket,
                    "symbol": d.symbol,
                    "type": "BUY" if d.type == 0 else "SELL", # Approximation
                    "volume": d.volume,
                    "price": d.price,
                    "profit": d.profit,
                    "time": datetime.fromtimestamp(d.time).strftime('%Y-%m-%d %H:%M'),
                    "exit_time": datetime.fromtimestamp(d.time) # For analytics object
                }
                
                # Check if it's a trade record vs balance op
                if d.symbol: 
                    temp_analytics.record_trade({"pnl": d.profit, "symbol": d.symbol, "exit_time": t["exit_time"]})
                    trades_list.append(t)
        
        # 3. Get Advanced Stats
        stats = temp_analytics.get_overall_stats()
        
        # Add extra derived metrics
        import numpy as np
        pnls = [t['profit'] for t in trades_list]
        
        # Clean up trades list for JSON serialization
        # remove raw datetime objects that are not serializable
        clean_trades = []
        for t in trades_list:
            clean_t = t.copy()
            if "exit_time" in clean_t:
                del clean_t["exit_time"]
            clean_trades.append(clean_t)

        return {
            "trades": list(reversed(clean_trades))[:50], # Last 50
            "metrics": stats,
            "daily_summary": temp_analytics.get_daily_summary(7)
        }
            
    except Exception as e:
        print(f"History error: {e}")
        return {"trades": [], "metrics": {}}


# --- HTML Frontend ---
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏦 Quant Fund Dashboard</title>
    <style>
        :root {
            --bg: #0b0e14; --panel: #151a25; --border: #2a3441;
            --text: #c9d1d9; --text-muted: #8b949e;
            --accent: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
            --font: 'Segoe UI', system-ui, sans-serif;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: var(--bg); color: var(--text); font-family: var(--font); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        
        /* Layout */
        header { 
            height: 60px; background: var(--panel); border-bottom: 1px solid var(--border); 
            display: flex; align-items: center; padding: 0 20px; justify-content: space-between;
        }
        .brand { font-size: 1.2rem; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 10px; }
        .brand span { color: var(--accent); }
        .server-status { font-size: 0.8rem; color: var(--green); display: flex; align-items: center; gap: 6px; }
        .server-status::before { content: ''; width: 8px; height: 8px; background: currentColor; border-radius: 50%; box-shadow: 0 0 8px currentColor; }
        
        /* Tabs */
        .tabs { display: flex; gap: 2px; }
        .tab { padding: 8px 16px; cursor: pointer; color: var(--text-muted); border-bottom: 2px solid transparent; transition: 0.2s; font-size: 0.9rem; }
        .tab:hover { color: var(--text); background: rgba(255,255,255,0.03); }
        .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
        
        main { flex: 1; overflow: hidden; position: relative; }
        .page { position: absolute; inset: 0; padding: 20px; overflow-y: auto; display: none; }
        .page.active { display: block; animation: fadein 0.3s; }
        @keyframes fadein { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

        /* Dashboard Grid */
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 15px; }
        .card h3 { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; }
        .card .value { font-size: 1.8rem; font-weight: 600; color: var(--text); }
        .card .sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 5px; }
        .text-green { color: var(--green) !important; }
        .text-red { color: var(--red) !important; }
        
        /* Workflow Visualization */
        .workflow-container { display: flex; flex-direction: column; gap: 20px; max-width: 1200px; margin: 0 auto; }
        .pipeline-track { display: flex; justify-content: space-between; position: relative; padding: 20px 0; }
        .pipeline-track::before { 
            content: ''; position: absolute; top: 35px; left: 0; right: 0; height: 2px; background: var(--border); z-index: 0; 
        }
        .step { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; width: 80px; text-align: center; }
        .step-icon { 
            width: 32px; height: 32px; background: var(--panel); border: 2px solid var(--border); border-radius: 50%; 
            display: flex; align-items: center; justify-content: center; font-size: 1.2rem; transition: 0.3s;
        }
        .step-label { font-size: 0.75rem; margin-top: 8px; color: var(--text-muted); font-weight: 500; }
        
        /* Step States */
        .step.running .step-icon { border-color: var(--accent); box-shadow: 0 0 15px rgba(88, 166, 255, 0.3); transform: scale(1.1); background: var(--panel); }
        .step.running .step-label { color: var(--accent); }
        .step.completed .step-icon { border-color: var(--green); color: var(--green); background: rgba(63, 185, 80, 0.1); }
        .step.error .step-icon { border-color: var(--red); color: var(--red); }
        .step.skipped .step-icon { border-color: var(--border); color: var(--text-muted); opacity: 0.5; }

        .logs-panel { background: #000; border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.8rem; height: 300px; overflow-y: auto; }
        .log-entry { margin-bottom: 4px; border-bottom: 1px solid #111; padding-bottom: 2px; display:flex; gap:10px; }
        .log-ts { color: var(--text-muted); min-width: 130px; }
        
        /* Project Tracker */
        .phase-list { display: flex; flex-direction: column; gap: 15px; max-width: 1000px; margin: 0 auto; }
        .phase-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
        .phase-header { padding: 15px; display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); cursor: pointer; }
        .phase-title { font-weight: 600; display: flex; align-items: center; gap: 10px; }
        .phase-status { font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; font-weight: 700; text-transform: uppercase; }
        .status-completed { background: rgba(63, 185, 80, 0.2); color: var(--green); }
        .status-in-progress { background: rgba(88, 166, 255, 0.2); color: var(--accent); }
        .status-pending { background: rgba(139, 148, 158, 0.2); color: var(--text-muted); }
        
        .progress-bar { height: 4px; background: var(--bg); width: 100%; position: relative; }
        .progress-fill { height: 100%; background: var(--green); transition: width 0.5s; }
        
        .phase-items { padding: 15px; display: none; border-top: 1px solid var(--border); }
        .phase-card.expanded .phase-items { display: block; }
        .phase-item { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 0.9rem; color: var(--text-muted); }
        .phase-item::before { content: '•'; color: var(--accent); }

        /* Responsive Table */
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
        th { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; }
        tr:hover { background: rgba(255,255,255,0.02); }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span>🚀</span> ANTIGRAVITY <span>DASHBOARD</span>
        </div>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview')">Overview</div>
            <div class="tab" onclick="setTab('workflow')">Live Workflow</div>
            <div class="tab" onclick="setTab('project')">Project Plan</div>
            <div class="tab" onclick="setTab('history')">History & Analytics</div>
        </div>
        <div class="server-status" id="connection-status">Disconnected</div>
    </header>

    <main>
        <!-- PAGE 1: OVERVIEW -->
        <div id="overview" class="page active">
            <div class="grid">
                <div class="card">
                    <h3>Balance</h3>
                    <div class="value" id="balance">$0.00</div>
                    <div class="sub" id="equity">Equity: $0.00</div>
                </div>
                <div class="card">
                    <h3>Profit (Session)</h3>
                    <div class="value" id="profit">$0.00</div>
                    <div class="sub">Drawdown: <span id="drawdown">0.00%</span></div>
                </div>
                <div class="card">
                    <h3>Active Trades</h3>
                    <div class="value" id="trade-count">0</div>
                    <div class="sub">Exposure: <span id="margin-level">0%</span></div>
                </div>
                <div class="card">
                    <h3>System State</h3>
                    <div class="value" style="font-size:1.2rem" id="sys-state">IDLE</div>
                    <div class="sub" id="last-update">-</div>
                </div>
            </div>

            <div class="card" style="margin-top: 20px;">
                <h3>Active Positions</h3>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Ticket</th>
                                <th>Symbol</th>
                                <th>Type</th>
                                <th>Volume</th>
                                <th>Pips</th>
                                <th>Profit</th>
                            </tr>
                        </thead>
                        <tbody id="positions-table">
                            <tr><td colspan="6" style="text-align:center; color: var(--text-muted)">No active positions</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- PAGE 2: WORKFLOW -->
        <div id="workflow" class="page">
            <div class="workflow-container">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="color: var(--accent);">AI Trading Pipeline</h2>
                    <p style="color: var(--text-muted); font-size: 0.9rem;">Real-time Execution Flow & State Monitoring</p>
                </div>
                
                <div class="card">
                    <div class="pipeline-track" id="pipeline-viz">
                        <!-- Pipeline Steps Injected Here -->
                    </div>
                </div>

                <div class="grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="card">
                        <h3>Current Action</h3>
                        <div class="value" style="font-size: 1.1rem; color: var(--accent);" id="wf-current-action">Waiting for cycle...</div>
                        <div class="sub" id="wf-details" style="margin-top: 10px; line-height: 1.4;">-</div>
                    </div>
                    <div class="card">
                        <h3>Cycle Metrics</h3>
                        <div class="sub">Active Symbol: <b id="wf-symbol" style="color:var(--text)">-</b></div>
                        <div class="sub">Last Pulse: <b id="wf-pulse" style="color:var(--text)">-</b></div>
                    </div>
                </div>

                <div class="card">
                    <h3>System Logs</h3>
                    <div class="logs-panel" id="wf-logs">
                        <!-- Logs injected here -->
                    </div>
                </div>
            </div>
        </div>

        <!-- PAGE 3: PROJECT -->
        <div id="project" class="page">
            <div class="phase-list" id="project-list">
                <!-- Project Phases Injected Here -->
            </div>
        </div>

        <!-- PAGE 4: HISTORY -->
        <div id="history" class="page">
            <div class="grid">
                <div class="card">
                    <h3>Total PnL</h3>
                    <div class="value" id="hist-pnl">$0.00</div>
                    <div class="sub">Profit Factor: <span id="hist-pf" style="color:var(--text)">0.0</span></div>
                </div>
                <div class="card">
                    <h3>Win Rate</h3>
                    <div class="value" id="hist-wr">0.0%</div>
                    <div class="sub" id="hist-counts">0W - 0L</div>
                </div>
                <div class="card">
                    <h3>Expectancy</h3>
                    <div class="value" id="hist-exp">$0.00</div>
                    <div class="sub">Avg Win/Loss: <span id="hist-avg">0/0</span></div>
                </div>
                <div class="card">
                    <h3>Total Trades</h3>
                    <div class="value" id="hist-total">0</div>
                    <div class="sub">Last 30 Days</div>
                </div>
            </div>

            <div class="card">
                <h3>Detailed Trade History (Last 50)</h3>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Ticket</th>
                                <th>Symbol</th>
                                <th>Type</th>
                                <th>Volume</th>
                                <th>Price</th>
                                <th>Profit</th>
                            </tr>
                        </thead>
                        <tbody id="history-table">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <script>
        // --- State ---
        let ws;
        
        // --- Tabs ---
        function setTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            
            const tab = [...document.querySelectorAll('.tab')].find(t => t.textContent.toLowerCase().includes(name.includes('workflow') ? 'workflow' : name));
            if(tab) tab.classList.add('active');
            
            document.getElementById(name).classList.add('active');
        }

        function togglePhase(el) {
            el.parentElement.classList.toggle('expanded');
        }

        // --- Updates ---
        function updateDashboard(data) {
            // Overview
            if(data.account) {
                document.getElementById('balance').textContent = '$' + data.account.balance.toFixed(2);
                document.getElementById('equity').textContent = 'Equity: $' + data.account.equity.toFixed(2);
                document.getElementById('profit').textContent = '$' + data.account.profit.toFixed(2);
                const profEl = document.getElementById('profit');
                profEl.className = 'value ' + (data.account.profit >= 0 ? 'text-green' : 'text-red');
                document.getElementById('drawdown').textContent = data.account.drawdown_pct + '%';
                document.getElementById('margin-level').textContent = data.account.margin_level + '%';
            }
            
            if(data.positions) {
                document.getElementById('trade-count').textContent = data.positions.length;
                const tbody = document.getElementById('positions-table');
                if(data.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted)">No active positions</td></tr>';
                } else {
                    tbody.innerHTML = data.positions.map(p => `
                        <tr>
                            <td>${p.ticket}</td>
                            <td>${p.symbol}</td>
                            <td><span class="${p.type=='BUY'?'text-green':'text-red'}">${p.type}</span></td>
                            <td>${p.volume}</td>
                            <td class="${p.pips>=0?'text-green':'text-red'}">${p.pips}</td>
                            <td class="${p.profit>=0?'text-green':'text-red'}">$${p.profit.toFixed(2)}</td>
                        </tr>
                    `).join('');
                }
            }

            // Workflow
            if(data.workflow) {
                const wf = data.workflow;
                document.getElementById('sys-state').textContent = wf.is_running ? 'RUNNING 🟢' : 'IDLE ⚪';
                document.getElementById('wf-symbol').textContent = wf.active_symbol || '-';
                document.getElementById('wf-pulse').textContent = new Date(wf.last_update).toLocaleTimeString();
                
                // Pipeline Viz
                const track = document.getElementById('pipeline-viz');
                track.innerHTML = wf.steps.map((step, i) => {
                    let cls = step.status.toLowerCase();
                    if(step.status === 'RUNNING') cls += ' running';
                    return `
                        <div class="step ${cls}">
                            <div class="step-icon" title="${step.status}\n${step.details}">${step.icon || '•'}</div>
                            <div class="step-label">${step.name}</div>
                        </div>
                    `;
                }).join('');

                // Details
                if(wf.current_step_index >= 0 && wf.steps[wf.current_step_index]) {
                    const s = wf.steps[wf.current_step_index];
                    document.getElementById('wf-current-action').textContent = s.name + ': ' + s.status;
                    document.getElementById('wf-details').textContent = s.details || '';
                } else {
                    document.getElementById('wf-current-action').textContent = 'Idle';
                    document.getElementById('wf-details').textContent = 'Waiting for next cycle...';
                }

                // Logs
                const logs = document.getElementById('wf-logs');
                logs.innerHTML = wf.logs.map(l => `
                    <div class="log-entry">
                        <span class="log-ts">[${l.ts.split('T')[1].split('.')[0]}]</span>
                        <span class="${l.level==='ERROR'?'text-red':'text-green'}">[${l.level}]</span>
                        <span>${l.message}</span>
                    </div>
                `).reverse().join('');
            }

            // Project (Update infrequently or on demand, but here we just update)
            if(data.project) {
                const list = document.getElementById('project-list');
                // Only render if empty to avoid closing details (simple optimization)
                if(list.children.length === 0 || data.project_update) {
                    list.innerHTML = data.project.map(p => `
                        <div class="phase-card">
                            <div class="phase-header" onclick="togglePhase(this)">
                                <div class="phase-title">${p.id}. ${p.name}</div>
                                <div class="phase-status status-${p.status.toLowerCase().replace('_','-')}">${p.status}</div>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${p.progress}%"></div>
                            </div>
                            <div class="phase-items">
                                ${p.items.map(i => `<div class="phase-item">${i}</div>`).join('')}
                            </div>
                        </div>
                    `).join('');
                }
            }

            // History
            if(data.history) {
                const m = data.history.metrics;
                const t = data.history.trades;
                
                if(m) {
                    const pnl = m.total_pnl || 0;
                    document.getElementById('hist-pnl').textContent = '$' + pnl.toFixed(2);
                    document.getElementById('hist-pnl').className = 'value ' + (pnl >= 0 ? 'text-green' : 'text-red');
                    
                    document.getElementById('hist-pf').textContent = (m.profit_factor || 0).toFixed(2);
                    
                    document.getElementById('hist-wr').textContent = ((m.win_rate || 0)*100).toFixed(1) + '%';
                    document.getElementById('hist-counts').textContent = `${m.winning_trades || 0}W - ${m.losing_trades || 0}L`;
                    
                    document.getElementById('hist-exp').textContent = '$' + (m.expectancy || 0).toFixed(2);
                    document.getElementById('hist-avg').textContent = `$${(m.avg_win||0).toFixed(0)} / $${Math.abs(m.avg_loss||0).toFixed(0)}`;
                    
                    document.getElementById('hist-total').textContent = m.total_trades || 0;
                }

                if(t) {
                     document.getElementById('history-table').innerHTML = t.map(row => `
                        <tr>
                            <td class="log-ts">${row.time}</td>
                            <td>${row.ticket}</td>
                            <td>${row.symbol}</td>
                            <td>${row.type}</td>
                            <td>${row.volume}</td>
                            <td>${row.price}</td>
                            <td class="${row.profit>=0?'text-green':'text-red'}">${row.profit>0?'+':''}${row.profit.toFixed(2)}</td>
                        </tr>
                     `).join('');
                }
            }
        }

        // --- WebSocket ---
        function connect() {
            const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${proto}://${window.location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').classList.remove('text-red');
                document.getElementById('connection-status').classList.add('text-green');
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                } catch(e) { console.error(e); }
            };
            
            ws.onclose = () => {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').classList.remove('text-green');
                document.getElementById('connection-status').classList.add('text-red');
                setTimeout(connect, 3000);
            };
        }

        // Start
        connect();
    </script>
</body>
</html>
"""

if HAS_FASTAPI:
    app = FastAPI(title="Hedge Fund Dashboard V2")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_CONTENT

    @app.get("/api/status")
    async def get_status():
        return {
            "account": get_mt5_account(),
            "positions": get_mt5_positions(),
            "workflow": workflow_state.get_state(),
            "project": project_tracker.get_status(),
            "history": get_detailed_history()
        }

    @app.post("/api/workflow/update")
    async def update_workflow(data: dict):
        workflow_state.update_from_client(data)
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = {
                    "account": get_mt5_account(),
                    "positions": get_mt5_positions(),
                    "workflow": workflow_state.get_state(),
                    "project": project_tracker.get_status(),
                    "history": get_detailed_history()
                }
                await websocket.send_json(data)
                await asyncio.sleep(1.0) # 1Hz update
        except WebSocketDisconnect:
            pass

def run_dashboard(host="0.0.0.0", port=8000):
    if not HAS_FASTAPI:
        print("Error: FastAPI not installed.")
        return
    
    print(f"🏦 Dashboard V2 running at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="error")

if __name__ == "__main__":
    run_dashboard()
