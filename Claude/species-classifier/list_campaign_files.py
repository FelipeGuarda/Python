"""List root-level files in the Primavera 2025 campaign directory."""
import os
import yaml
from pathlib import Path

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

campaign = Path(config["campaign_dir"])
print(f"Campaign dir: {campaign}\n")

files = sorted([f for f in campaign.iterdir() if f.is_file()], key=lambda f: f.name)
print(f"{'File':<55} {'Size':>10}  {'Modified'}")
print("-" * 90)
for f in files:
    stat = f.stat()
    size = stat.st_size
    mtime = os.path.getmtime(f)
    import datetime
    dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    size_str = f"{size:,}" if size < 1_000_000 else f"{size/1_000_000:.1f} MB"
    print(f"{f.name:<55} {size_str:>10}  {dt}")
