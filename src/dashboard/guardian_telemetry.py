# src/dashboard/guardian_telemetry.py
"""
Guardian Telemetry Server
==========================

UDP-based telemetry for live dashboard streaming.
MT5/live_loop sends → Server buffers → Streamlit polls

Usage:
    # Start server
    python -m src.dashboard.guardian_telemetry

    # From live_loop
    send_telemetry({"dd": 0.03, "margin": 75, ...})
"""

import socket
import json
import threading
import queue
from datetime import datetime
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("TELEMETRY")

HOST = "127.0.0.1"
PORT = 9999

# Thread-safe queue for telemetry data
telemetry_queue = queue.Queue(maxsize=10000)
_server_running = False


def start_server(host: str = HOST, port: int = PORT):
    """Start UDP telemetry server."""
    global _server_running
    
    if _server_running:
        return
    
    def server_loop():
        global _server_running
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(1.0)
        
        logger.info(f"🛡️ Telemetry Server started on {host}:{port}")
        _server_running = True
        
        while _server_running:
            try:
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode())
                msg["_received"] = datetime.now().isoformat()
                
                if not telemetry_queue.full():
                    telemetry_queue.put(msg)
            except socket.timeout:
                continue
            except Exception as e:
                logger.debug(f"Telemetry error: {e}")
        
        sock.close()
    
    thread = threading.Thread(target=server_loop, daemon=True)
    thread.start()


def stop_server():
    """Stop telemetry server."""
    global _server_running
    _server_running = False


def get_latest(count: int = 100) -> list:
    """Get latest telemetry messages."""
    messages = []
    try:
        while len(messages) < count and not telemetry_queue.empty():
            messages.append(telemetry_queue.get_nowait())
    except:
        pass
    return messages


def get_all() -> list:
    """Drain all messages from queue."""
    messages = []
    while not telemetry_queue.empty():
        try:
            messages.append(telemetry_queue.get_nowait())
        except:
            break
    return messages


# =============================================================================
# Client: Send from live_loop
# =============================================================================

_client_sock: Optional[socket.socket] = None


def send_telemetry(data: Dict, host: str = HOST, port: int = PORT):
    """
    Send telemetry from live_loop.
    
    Args:
        data: Telemetry payload dict
    """
    global _client_sock
    
    try:
        if _client_sock is None:
            _client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        data["ts"] = datetime.now().isoformat()
        payload = json.dumps(data).encode()
        _client_sock.sendto(payload, (host, port))
    except Exception as e:
        logger.debug(f"Telemetry send error: {e}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("\n🛡️ Starting Guardian Telemetry Server...")
    start_server()
    
    print(f"   Listening on UDP {HOST}:{PORT}")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            msgs = get_all()
            for m in msgs:
                print(f"  📡 {m}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_server()
        print("\n✅ Server stopped")
