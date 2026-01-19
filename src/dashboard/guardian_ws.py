# src/dashboard/guardian_ws.py
"""
Guardian Dashboard WebSocket
=============================

Realtime streaming of Guardian vs Alpha metrics.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_WS")


@dataclass
class GuardianDashboardState:
    """Current state for dashboard."""
    # Alpha Agent
    alpha_signal: str = "HOLD"
    alpha_confidence: float = 0.0
    open_positions: int = 0
    unrealized_pnl: float = 0.0
    
    # Guardian Agent
    guardian_action: str = "ALLOW"
    margin_ratio: float = 1.0
    daily_dd: float = 0.0
    freeze_timer: int = 0
    escalation_level: int = 0
    block_count: int = 0
    
    # Account
    equity: float = 0.0
    balance: float = 0.0
    
    # Metadata
    timestamp: str = ""
    cycle: int = 0


class GuardianEventLog:
    """Event log for Guardian timeline."""
    
    def __init__(self, max_events: int = 100):
        self.events: List[Dict] = []
        self.max_events = max_events
    
    def add_event(self, event_type: str, message: str, severity: str = "INFO"):
        """Add event to timeline."""
        event = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "message": message,
            "severity": severity,
        }
        self.events.append(event)
        
        # Trim if too many
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_recent(self, count: int = 20) -> List[Dict]:
        """Get recent events."""
        return self.events[-count:]


# Global state (updated by live loop)
_dashboard_state = GuardianDashboardState()
_event_log = GuardianEventLog()


def update_dashboard_state(
    alpha_signal: str = None,
    alpha_confidence: float = None,
    guardian_action: str = None,
    margin_ratio: float = None,
    daily_dd: float = None,
    freeze_timer: int = None,
    escalation_level: int = None,
    block_count: int = None,
    equity: float = None,
    balance: float = None,
    open_positions: int = None,
    unrealized_pnl: float = None,
    cycle: int = None,
):
    """Update dashboard state from live loop."""
    global _dashboard_state
    
    if alpha_signal is not None:
        _dashboard_state.alpha_signal = alpha_signal
    if alpha_confidence is not None:
        _dashboard_state.alpha_confidence = alpha_confidence
    if guardian_action is not None:
        _dashboard_state.guardian_action = guardian_action
    if margin_ratio is not None:
        _dashboard_state.margin_ratio = margin_ratio
    if daily_dd is not None:
        _dashboard_state.daily_dd = daily_dd
    if freeze_timer is not None:
        _dashboard_state.freeze_timer = freeze_timer
    if escalation_level is not None:
        _dashboard_state.escalation_level = escalation_level
    if block_count is not None:
        _dashboard_state.block_count = block_count
    if equity is not None:
        _dashboard_state.equity = equity
    if balance is not None:
        _dashboard_state.balance = balance
    if open_positions is not None:
        _dashboard_state.open_positions = open_positions
    if unrealized_pnl is not None:
        _dashboard_state.unrealized_pnl = unrealized_pnl
    if cycle is not None:
        _dashboard_state.cycle = cycle
    
    _dashboard_state.timestamp = datetime.now().isoformat()


def add_guardian_event(event_type: str, message: str, severity: str = "INFO"):
    """Add event to Guardian timeline."""
    global _event_log
    _event_log.add_event(event_type, message, severity)


def get_dashboard_state() -> Dict:
    """Get current dashboard state as dict."""
    return asdict(_dashboard_state)


def get_event_timeline(count: int = 20) -> List[Dict]:
    """Get recent events."""
    return _event_log.get_recent(count)


# =============================================================================
# FastAPI WebSocket (if fastapi available)
# =============================================================================

def create_guardian_websocket_app():
    """Create FastAPI app with Guardian WebSocket."""
    try:
        from fastapi import FastAPI, WebSocket
        from fastapi.responses import HTMLResponse
    except ImportError:
        logger.warning("FastAPI not available for WebSocket")
        return None
    
    app = FastAPI(title="Guardian Dashboard WS")
    
    # Dashboard HTML
    DASHBOARD_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Guardian Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
            .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1200px; margin: 0 auto; }
            .card { background: #16213e; border-radius: 12px; padding: 20px; }
            .card h3 { margin-top: 0; color: #00d9ff; border-bottom: 1px solid #333; padding-bottom: 10px; }
            .metric { display: flex; justify-content: space-between; margin: 10px 0; }
            .metric-value { font-weight: bold; font-size: 1.2em; }
            .alpha { color: #4ecca3; }
            .guardian { color: #ff6b6b; }
            .timeline { background: #0f0f23; border-radius: 8px; padding: 15px; max-height: 300px; overflow-y: auto; }
            .event { padding: 5px 0; border-bottom: 1px solid #222; font-size: 0.9em; }
            .event-time { color: #666; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #00d9ff; margin-bottom: 5px; }
            .status-bar { display: flex; justify-content: center; gap: 40px; margin-top: 20px; }
            .status-item { text-align: center; }
            .status-value { font-size: 2em; font-weight: bold; }
            .ALLOW { color: #4ecca3; }
            .REDUCE_RISK { color: #ffd93d; }
            .FORCE_HOLD { color: #ff9f43; }
            .EMERGENCY_FREEZE { color: #ff6b6b; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ Guardian Dashboard</h1>
            <div class="status-bar">
                <div class="status-item">
                    <div>Equity</div>
                    <div class="status-value" id="equity">$0.00</div>
                </div>
                <div class="status-item">
                    <div>Daily DD</div>
                    <div class="status-value" id="dd">0.0%</div>
                </div>
                <div class="status-item">
                    <div>Margin</div>
                    <div class="status-value" id="margin">100%</div>
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h3 class="alpha">🎯 Alpha Agent</h3>
                <div class="metric">
                    <span>Signal</span>
                    <span class="metric-value" id="alpha-signal">HOLD</span>
                </div>
                <div class="metric">
                    <span>Confidence</span>
                    <span class="metric-value" id="alpha-confidence">0%</span>
                </div>
                <div class="metric">
                    <span>Open Positions</span>
                    <span class="metric-value" id="positions">0</span>
                </div>
                <div class="metric">
                    <span>Unrealized P&L</span>
                    <span class="metric-value" id="pnl">$0.00</span>
                </div>
            </div>
            
            <div class="card">
                <h3 class="guardian">🛡️ Guardian Agent</h3>
                <div class="metric">
                    <span>Action</span>
                    <span class="metric-value" id="guardian-action">ALLOW</span>
                </div>
                <div class="metric">
                    <span>Escalation Level</span>
                    <span class="metric-value" id="escalation">0</span>
                </div>
                <div class="metric">
                    <span>Block Count</span>
                    <span class="metric-value" id="blocks">0</span>
                </div>
                <div class="metric">
                    <span>Freeze Timer</span>
                    <span class="metric-value" id="freeze">0s</span>
                </div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px; max-width: 1160px; margin-left: auto; margin-right: auto;">
            <h3>📋 Guardian Event Timeline</h3>
            <div class="timeline" id="timeline">
                <div class="event"><span class="event-time">--:--:--</span> Waiting for events...</div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://' + window.location.host + '/ws/guardian');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Update metrics
                document.getElementById('equity').textContent = '$' + data.state.equity.toFixed(2);
                document.getElementById('dd').textContent = (data.state.daily_dd * 100).toFixed(1) + '%';
                document.getElementById('margin').textContent = (data.state.margin_ratio * 100).toFixed(0) + '%';
                
                document.getElementById('alpha-signal').textContent = data.state.alpha_signal;
                document.getElementById('alpha-confidence').textContent = (data.state.alpha_confidence * 100).toFixed(0) + '%';
                document.getElementById('positions').textContent = data.state.open_positions;
                document.getElementById('pnl').textContent = '$' + data.state.unrealized_pnl.toFixed(2);
                
                const actionEl = document.getElementById('guardian-action');
                actionEl.textContent = data.state.guardian_action;
                actionEl.className = 'metric-value ' + data.state.guardian_action;
                
                document.getElementById('escalation').textContent = data.state.escalation_level;
                document.getElementById('blocks').textContent = data.state.block_count;
                document.getElementById('freeze').textContent = data.state.freeze_timer + 's';
                
                // Update timeline
                if (data.events && data.events.length > 0) {
                    const timeline = document.getElementById('timeline');
                    timeline.innerHTML = data.events.map(e => 
                        '<div class="event"><span class="event-time">' + e.timestamp + '</span> ' + e.message + '</div>'
                    ).join('');
                }
            };
        </script>
    </body>
    </html>
    """
    
    @app.get("/")
    async def get_dashboard():
        return HTMLResponse(DASHBOARD_HTML)
    
    @app.websocket("/ws/guardian")
    async def guardian_websocket(websocket: WebSocket):
        await websocket.accept()
        logger.info("Guardian WebSocket client connected")
        
        try:
            while True:
                # Send current state
                await websocket.send_json({
                    "state": get_dashboard_state(),
                    "events": get_event_timeline(20),
                })
                await asyncio.sleep(1)
        except Exception as e:
            logger.debug(f"WebSocket closed: {e}")
    
    return app


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    app = create_guardian_websocket_app()
    if app:
        print("\n🛡️ Starting Guardian Dashboard...")
        print("   Open http://localhost:8080 in your browser\n")
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
