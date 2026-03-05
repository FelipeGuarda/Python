"""
Quick analysis of CLIP confidence score distribution across classified images.
"""
import csv
from collections import defaultdict
from pathlib import Path
import yaml

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

csv_path = Path(config["campaign_dir"]) / config["output_csv"]

rows = []
with open(csv_path, encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["observationType"] == "animal"]

scores = [float(r["classificationProbability"]) for r in rows]
scores.sort()

n = len(scores)
print(f"Total classified images: {n}")
print()

# Overall distribution
thresholds = [0.10, 0.15, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.35]
print(f"{'Threshold':<12} {'Below (reject)':<18} {'Above (keep)':<15} {'% kept'}")
print("-" * 58)
for t in thresholds:
    below = sum(1 for s in scores if s < t)
    above = n - below
    print(f"  {t:<10.2f} {below:<18} {above:<15} {100*above/n:.1f}%")

print()

# Per-species breakdown
by_species = defaultdict(list)
for r in rows:
    by_species[r["observationComments"]].append(float(r["classificationProbability"]))

print(f"{'Species':<25} {'N':>4}  {'Min':>6}  {'Median':>6}  {'Max':>6}  {'<0.22':>6}")
print("-" * 62)
for sp, sp_scores in sorted(by_species.items(), key=lambda x: -len(x[1])):
    sp_scores_s = sorted(sp_scores)
    med = sp_scores_s[len(sp_scores_s)//2]
    low = sum(1 for s in sp_scores if s < 0.22)
    print(f"{sp:<25} {len(sp_scores):>4}  {min(sp_scores):>6.3f}  {med:>6.3f}  {max(sp_scores):>6.3f}  {low:>6}")

print()
# Percentiles
print("Score percentiles:")
for pct in [5, 10, 15, 20, 25, 50]:
    idx = int(pct / 100 * n)
    print(f"  p{pct:02d}: {scores[idx]:.4f}")
