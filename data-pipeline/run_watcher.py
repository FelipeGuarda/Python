import os
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.db import connect, init_schema
from src.watcher import start_watcher

with open("config.yaml") as f:
    config = yaml.safe_load(f)

incoming_dir = Path(config["watcher"]["incoming_dir"])

# One-shot bootstrap: ensure the schema exists, then release the DB lock.
# Each filesystem-event handler opens its own short-lived connection.
_bootstrap = connect()
try:
    init_schema(_bootstrap)
finally:
    _bootstrap.close()

observer = start_watcher(incoming_dir, connect)

try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    print("\n→ Watcher stopped.")
finally:
    observer.stop()
    observer.join()
