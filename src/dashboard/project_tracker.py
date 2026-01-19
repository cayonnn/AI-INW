import json
import os
from datetime import datetime

class ProjectTracker:
    def __init__(self, storage_path="project_status.json"):
        self.storage_path = storage_path
        self.phases = self._load_or_default()

    def _default_phases(self):
        return [
            {"id": 1, "name": "Research & Strategy", "status": "COMPLETED", "progress": 100, "items": ["Market Hypothesis", "Signal Candidates", "Dataset Construction"]},
            {"id": 2, "name": "Model Development", "status": "COMPLETED", "progress": 100, "items": ["XGBoost Model", "LSTM Model", "Offline Training"]},
            {"id": 3, "name": "EA Architecture", "status": "COMPLETED", "progress": 100, "items": ["MQL5 Modules", "DataFeed", "TradeExecutor"]},
            {"id": 4, "name": "Signal Fusion Logic", "status": "IN_PROGRESS", "progress": 80, "items": ["Rule Signals", "AI Probabilities", "Decision Gate"]},
            {"id": 5, "name": "Risk Management", "status": "IN_PROGRESS", "progress": 90, "items": ["Position Sizing", "Guardian Latch", "Stop Loss"]},
            {"id": 6, "name": "Backtest & Validation", "status": "PENDING", "progress": 30, "items": ["Single Symbol", "Multi-symbol", "Monte Carlo"]},
            {"id": 7, "name": "Deployment", "status": "PENDING", "progress": 0, "items": ["Demo Account", "Small Live", "Scale"]},
            {"id": 8, "name": "Live Monitoring", "status": "PENDING", "progress": 10, "items": ["Metrics Tracking", "Drift Detection"]}
        ]

    def _load_or_default(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return self._default_phases()
        return self._default_phases()

    def save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.phases, f, indent=2)

    def get_status(self):
        return self.phases

    def update_phase(self, phase_id, status, progress):
        for p in self.phases:
            if p["id"] == phase_id:
                p["status"] = status
                p["progress"] = progress
                p["last_updated"] = datetime.now().isoformat()
        self.save()
