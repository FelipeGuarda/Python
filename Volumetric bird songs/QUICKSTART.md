# Quick Start Guide

## 1. Activate Environment

```bash
conda activate bird_viz
```

## 2. Run with Your Audio File

```bash
# Basic usage - creates all visualizations
python volumetric_bird_songs.py path/to/your/birdsong.wav

# Example with a file in sample_audio directory
python volumetric_bird_songs.py sample_audio/my_bird.wav
```

## 3. View Results

Open the generated HTML files in your browser:
- `examples/polar_animation.html` - **Start here!** Animated circular visualization
- `examples/diamond_3d_mirror.html` - 3D diamond with mirrored features
- `examples/diamond_3d_multi.html` - 3D diamond with four features
- `examples/comparison_dashboard.html` - Complete analysis dashboard

## Example Output Location

After running the script, you'll find:
```
examples/
├── polar_animation.html          ← Interactive animation (PRIMARY)
├── diamond_3d_mirror.html        ← 3D visualization (symmetric)
├── diamond_3d_multi.html         ← 3D visualization (asymmetric)
└── comparison_dashboard.html     ← Full analysis dashboard
```

## Need Help?

```bash
python volumetric_bird_songs.py --help
```

See `README.md` for detailed documentation.

## What Do the Visualizations Show?

### Polar Animation
- **Circle oscillation** = Frequency/pitch changes
- **Circle size** = Amplitude/loudness
- **Color** = Spectral centroid (brightness of sound)

### 3D Diamonds
- **Y-axis** = Time progression
- **X-axis** = Frequency features
- **Z-axis** = Amplitude features
- **Color** = Additional acoustic feature

### Dashboard
- Time series of all features
- Spectrogram heatmap
- Statistical summary

## Tips

1. **Start with the polar animation** - it's the most intuitive
2. Use the play/pause buttons and slider to explore
3. Try different bird calls to see how they differ visually
4. Rotate the 3D diamonds to see different perspectives

Happy visualizing! 🐦🎵
