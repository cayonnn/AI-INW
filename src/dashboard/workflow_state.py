from datetime import datetime
from collections import deque
import json
import requests
import threading
import queue

# ==========================================
# SERVER SIDE (State Store)
# ==========================================
class WorkflowState:
    def __init__(self):
        self.steps = [
            {"id": "init", "name": "Initialization", "status": "IDLE", "icon": "🚀"},
            {"id": "data", "name": "Market Data", "status": "IDLE", "icon": "📊"},
            {"id": "regime", "name": "Global Regime", "status": "IDLE", "icon": "🌍"},
            {"id": "committee", "name": "Risk Committee", "status": "IDLE", "icon": "🏛️"},
            {"id": "protection", "name": "Capital Protection", "status": "IDLE", "icon": "🛡️"},
            {"id": "features", "name": "Feature Engine", "status": "IDLE", "icon": "📐"},
            {"id": "ai", "name": "AI Inference", "status": "IDLE", "icon": "🧠"},
            {"id": "fusion", "name": "Signal Fusion", "status": "IDLE", "icon": "⚗️"},
            {"id": "gate", "name": "Decision Gate", "status": "IDLE", "icon": "🚪"},
            {"id": "risk", "name": "Position Sizing", "status": "IDLE", "icon": "⚖️"},
            {"id": "execution", "name": "Execution", "status": "IDLE", "icon": "⚡"},
            {"id": "trailing", "name": "Trailing Mgmt", "status": "IDLE", "icon": "📉"},
            {"id": "analytics", "name": "Post-Trade Analytics", "status": "IDLE", "icon": "📋"}
        ]
        self.step_map = {s["id"]: i for i, s in enumerate(self.steps)}
        self.current_step_index = -1
        self.logs = deque(maxlen=100)
        self.last_update = datetime.now().isoformat()
        self.is_running = False
        self.active_symbol = ""

    def update_from_client(self, data):
        """Update state from client payload."""
        if "active_symbol" in data:
            self.active_symbol = data["active_symbol"]
        
        if "is_running" in data:
            self.is_running = data["is_running"]

        if "update_step" in data:
            s = data["update_step"]
            self._update_step_internal(s["id"], s["status"], s.get("details"))

        if "log" in data:
            l = data["log"]
            self.add_log(l["level"], l["message"])

    def _update_step_internal(self, step_id, status, details=None):
        if step_id == "RESET":
            self.reset()
            return

        if step_id not in self.step_map:
            return

        idx = self.step_map[step_id]
        step = self.steps[idx]
        step["status"] = status
        if details:
            step["details"] = details

        if status == "RUNNING":
            self.current_step_index = idx
        
        self.last_update = datetime.now().isoformat()

    def reset(self):
        for step in self.steps:
            step["status"] = "IDLE"
            step["details"] = "-"
        self.current_step_index = -1
        self.is_running = False
        self.last_update = datetime.now().isoformat()

    def add_log(self, level, message):
        self.logs.append({
            "ts": datetime.now().isoformat(),
            "level": level,
            "message": message
        })

    def get_state(self):
        return {
            "steps": self.steps,
            "is_running": self.is_running,
            "active_symbol": self.active_symbol,
            "current_step_index": self.current_step_index,
            "last_update": self.last_update,
            "logs": list(self.logs)[-20:]
        }

# ==========================================
# CLIENT SIDE (Streamer) - FIXED: Queue + Worker
# ==========================================
class WorkflowClient:
    def __init__(self, host="http://localhost:8000"):
        self.host = host
        self.session = requests.Session()
        self._queue = queue.Queue(maxsize=100)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        """Single worker thread consumes queue."""
        while True:
            try:
                payload = self._queue.get(timeout=1)
                if payload is None:
                    break
                self.session.post(f"{self.host}/api/workflow/update", json=payload, timeout=0.5)
            except queue.Empty:
                pass
            except:
                pass  # Fire and forget

    def _send(self, payload):
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass  # Drop if queue full

    def start_cycle(self, symbol="XAUUSD"):
        self._send({
            "active_symbol": symbol,
            "is_running": True,
            "update_step": {"id": "RESET", "status": "IDLE"}
        })
        self.update_step("init", "RUNNING", "Starting cycle...")

    def update_step(self, step_id, status, details=None):
        self._send({
            "update_step": {"id": step_id, "status": status, "details": details}
        })

    def add_log(self, level, message):
        self._send({
            "log": {"level": level, "message": message}
        })

# Singleton Store
_server_state = WorkflowState()

def get_workflow_state():
    """For Server: Returns the state store."""
    return _server_state

def get_workflow_client():
    """For Client: Returns the HTTP streamer."""
    return WorkflowClient()
