#!/usr/bin/env python3
"""
Test script to verify the bird_viz environment is set up correctly.
Run this after activating the conda environment: conda activate bird_viz
"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'pandas': 'Pandas',
        'librosa': 'librosa',
        'plotly': 'Plotly',
        'soundfile': 'soundfile'
    }
    
    failed = []
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("\nMake sure you activated the conda environment:")
        print("  conda activate bird_viz")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_librosa_features():
    """Test that librosa can extract features from synthetic audio"""
    print("\nTesting audio feature extraction...")
    
    try:
        import numpy as np
        import librosa
        
        # Create synthetic audio (1 second, 440 Hz sine wave)
        sr = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Test feature extraction
        rms = librosa.feature.rms(y=audio)[0]
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        print(f"  Synthetic audio: {duration}s at {frequency}Hz")
        print(f"  RMS amplitude: {rms.mean():.4f}")
        print(f"  Spectral centroid: {centroid.mean():.1f} Hz")
        print("✓ Feature extraction works!")
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

def test_plotly():
    """Test that Plotly can create figures"""
    print("\nTesting Plotly visualization...")
    
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])
        print("✓ Plotly figure creation works!")
        return True
        
    except Exception as e:
        print(f"✗ Plotly test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Bird Song Visualizer - Installation Test")
    print("="*60)
    print()
    
    tests = [
        test_imports(),
        test_librosa_features(),
        test_plotly()
    ]
    
    print()
    print("="*60)
    
    if all(tests):
        print("✅ All tests passed! Installation is complete.")
        print()
        print("Next steps:")
        print("1. Place bird song audio files in sample_audio/")
        print("2. Run: python volumetric_bird_songs.py sample_audio/your_file.wav")
        print("3. Open the generated HTML files in examples/")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        print()
        print("Try reinstalling the environment:")
        print("  conda env remove -n bird_viz")
        print("  conda env create -f environment.yml")
        return 1

if __name__ == '__main__':
    sys.exit(main())
