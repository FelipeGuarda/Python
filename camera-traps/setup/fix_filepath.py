"""Fix empty filePath column in Otoño 2025 DB and CSV."""
import csv
import sqlite3
from pathlib import Path

CAMPAIGN = Path(r"C:\Users\USUARIO\SynologyDrive\2. Camaras trampa (SC)"
                r"\SynologyDrive\DATOS_GRILLA CÁMARAS TRAMPA"
                r"\2. CAMPAÑAS DE RECOLECCION DE IMAGENES\Otoño 2025\Fotos")

# 1. Fix DB
db = CAMPAIGN / "TimelapseData.ddb"
conn = sqlite3.connect(db)
conn.execute(r"UPDATE DataTable SET filePath = RelativePath || '\' || File "
             r"WHERE filePath IS NULL OR filePath = ''")
conn.commit()
updated = conn.execute("SELECT COUNT(*) FROM DataTable WHERE filePath != ''").fetchone()[0]
samples = conn.execute("SELECT filePath FROM DataTable LIMIT 3").fetchall()
conn.close()
print(f"DB: {updated} rows with filePath")
for s in samples:
    print(f"  {s[0]}")

# 2. Fix CSV
csv_path = CAMPAIGN / "ImageData_animals.csv"
with open(csv_path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

fixed = 0
for row in rows:
    if not row.get("filePath") and row.get("RelativePath") and row.get("File"):
        row["filePath"] = row["RelativePath"] + "\\" + row["File"]
        fixed += 1

with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV: {fixed} rows fixed")
for r in rows[:3]:
    print(f"  {r['filePath']}")
