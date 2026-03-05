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

con = connect()
init_schema(con)

observer = start_watcher(incoming_dir, con)

try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    observer.stop()
    print("\n→ Watcher stopped.")

observer.join()
