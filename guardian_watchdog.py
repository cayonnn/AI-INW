
import subprocess
import time
import sys
import logging
from datetime import datetime
import signal
import os

# Create log directory if not exists
os.makedirs("logs", exist_ok=True)

# Force UTF-8 for Windows Console (Fix Emoji Crash)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
SCRIPT_TO_MONITOR = "live_loop_v3.py"
RESTART_DELAY = 5  # Seconds
MAX_RESTARTS_PER_HOUR = 20

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [WATCHDOG] | %(message)s",
    handlers=[
        logging.FileHandler("logs/watchdog.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class Watchdog:
    def __init__(self):
        self.process = None
        self.running = True
        self.restart_counts = [] # timestamps of restarts
        
        # Handle cleanup on exit
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        """Clean shutdown."""
        logging.info("[STOP] Watchdog shutting down...")
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        sys.exit(0)

    def prune_restarts(self):
        """Remove restart timestamps older than 1 hour."""
        now = time.time()
        self.restart_counts = [t for t in self.restart_counts if now - t < 3600]

    def run(self):
        """Main loop."""
        logging.info(f"[START] Watchdog started. Monitoring: {SCRIPT_TO_MONITOR}")
        
        while self.running:
            try:
                # 1. Start Process
                logging.info("[RUN] Starting Live Loop...")
                start_time = time.time()
                
                # Using sys.executable to use the same python interpreter
                cmd = [sys.executable, SCRIPT_TO_MONITOR, "--live"]
                
                # Prune old heartbeat before start to prevent immediate kill
                if os.path.exists("heartbeat.txt"):
                    try:
                        os.remove("heartbeat.txt")
                    except Exception as e:
                        logging.warning(f"Could not remove old heartbeat: {e}")
                
                self.process = subprocess.Popen(
                    cmd,
                    cwd=os.getcwd()
                    # stdout/stderr default to parent (console)
                )
                
                # 2. Monitor Loop (Heartbeat & Exit)
                while True:
                    return_code = self.process.poll()
                    if return_code is not None:
                        break # Process exited
                    
                    # Heartbeat Check
                    try:
                        if os.path.exists("heartbeat.txt"):
                            with open("heartbeat.txt", "r") as f:
                                last_beat = float(f.read().strip())
                                if time.time() - last_beat > 60: # 60s freeze limit
                                    logging.error(f"[FREEZE] Process frozen (last heartbeat {time.time() - last_beat:.1f}s ago). Killing...")
                                    self.process.kill()
                                    self.process.wait()
                                    break
                    except Exception as e:
                        logging.warning(f"[WARN] Heartbeat read failed: {e}")
                        
                    time.sleep(5)
                
                # 3. Handle Exit
                runtime = time.time() - start_time
                
                if not self.running:
                    break
                    
                if return_code == 0:
                    logging.info(f"[OK] Live Loop exited normally (Runtime: {runtime:.1f}s). Restarting...")
                elif return_code is None: # Killed by us
                     logging.warning(f"[RESTART] Watchdog forced restart (Runtime: {runtime:.1f}s).")
                else:
                    logging.error(f"[WARN] Live Loop CRASHED (Code: {return_code}, Runtime: {runtime:.1f}s)")
                
                # 4. Check Restart Limit (Anti-Flap)
                self.restart_counts.append(time.time())
                self.prune_restarts()
                
                if len(self.restart_counts) > MAX_RESTARTS_PER_HOUR:
                    logging.critical(f"[CRITICAL] TOO MANY CRASHES ({len(self.restart_counts)}/hr). Watchdog giving up.")
                    break
                
                # 5. Cooldown
                logging.info(f"[WAIT] Restarting in {RESTART_DELAY}s...")
                time.sleep(RESTART_DELAY)
                
            except Exception as e:
                logging.critical(f"[CRITICAL] Watchdog Error: {e}")
                time.sleep(RESTART_DELAY)

if __name__ == "__main__":
    dog = Watchdog()
    dog.run()
