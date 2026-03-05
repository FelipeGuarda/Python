"""Delete obsolete files from the Primavera 2025 campaign directory."""
import yaml
from pathlib import Path

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

campaign = Path(config["C:\\Users\\USUARIO\\SynologyDrive\\2. Camaras trampa (SC)\\SynologyDrive\\DATOS_GRILLA CÁMARAS TRAMPA\\2. CAMPAÑAS DE RECOLECCION DE IMAGENES\\Primavera 2025"])

to_delete = [
    "image_recognition_file.json",           # wrong tropical model output
    "image_recognition_file_original.json",  # wrong tropical model original
    "model_special_char_log.txt",            # log from wrong model run
    "model_warning_log.txt",                 # log from wrong model run
    "results.xlsx",                          # results from wrong model
    "results_detections.csv",               # results from wrong model
    "results_files.csv",                    # results from wrong model
    "flatten_log_20260226_162931_dryrun.csv", # dry-run preprocessing log
    "flatten_log_20260226_163046.csv",        # preprocessing log
]

for name in to_delete:
    f = campaign / name
    if f.exists():
        f.unlink()
        print(f"  deleted: {name}")
    else:
        print(f"  not found: {name}")

print("Done.")
