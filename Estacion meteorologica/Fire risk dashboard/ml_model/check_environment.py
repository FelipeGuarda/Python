#!/usr/bin/env python
"""
Quick environment check before running ML training.
Verifies all required dependencies are installed.
"""

import sys
from pathlib import Path

def check_import(module_name: str, package_name: str = None) -> bool:
    """Try importing a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    print()
    
    # Check required packages
    print("Required packages:")
    all_ok = True
    
    packages = [
        ("pandas", None),
        ("numpy", None),
        ("requests", None),
        ("sklearn", "scikit-learn"),
        ("joblib", None),
        ("matplotlib", None),
        ("pyDataverse", None),
    ]
    
    for module, package in packages:
        if not check_import(module, package):
            all_ok = False
    
    print()
    
    # Check file structure
    print("File structure:")
    base_dir = Path(__file__).parent.parent
    ml_dir = Path(__file__).parent
    
    checks = [
        (ml_dir / "prepare_training_data.py", "prepare_training_data.py"),
        (ml_dir / "train_fire_model.py", "train_fire_model.py"),
        (base_dir / "download_dataverse.py", "download_dataverse.py"),
        (ml_dir / "data", "ml_model/data/ directory"),
    ]
    
    for path, name in checks:
        if path.exists():
            print(f"✓ {name}")
        else:
            print(f"✗ {name} - NOT FOUND")
            all_ok = False
    
    print()
    
    # Check if fire data is downloaded
    fire_data = ml_dir / "data" / "cicatrices_incendios_resumen.geojson"
    if fire_data.exists():
        size_mb = fire_data.stat().st_size / 1024 / 1024
        print(f"✓ Fire dataset downloaded ({size_mb:.1f} MB)")
    else:
        print("⚠️  Fire dataset NOT downloaded yet")
        print("   Run: ./download_fire_data.sh YOUR_API_KEY")
    
    print()
    
    # Check MAX_SAMPLES setting
    prepare_script = ml_dir / "prepare_training_data.py"
    if prepare_script.exists():
        content = prepare_script.read_text()
        if "MAX_SAMPLES = None" in content:
            print("✓ MAX_SAMPLES configured for full training (None)")
        elif "MAX_SAMPLES = 50" in content:
            print("⚠️  MAX_SAMPLES still set to 50 (will only use 50 fires)")
            print("   Should be: MAX_SAMPLES = None")
        else:
            print("? MAX_SAMPLES setting unclear")
    
    print()
    print("="*60)
    
    if all_ok:
        print("STATUS: ✓ Ready to train!")
        if not fire_data.exists():
            print()
            print("Next step: Download fire dataset")
            print("  Run: ./download_fire_data.sh YOUR_API_KEY")
        else:
            print()
            print("Next step: Prepare training data")
            print("  Run: python prepare_training_data.py")
    else:
        print("STATUS: ✗ Some issues found - fix them first")
        print()
        print("To fix missing packages:")
        print("  conda activate fire_risk_dashboard")
        print("  conda install scikit-learn pandas numpy matplotlib")
        print("  pip install pyDataverse==0.3.3")
    
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
