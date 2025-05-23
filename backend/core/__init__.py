from datetime import datetime
from pathlib import Path

def __init__(self):
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    self.simulation_id = str(datetime.now().timestamp)