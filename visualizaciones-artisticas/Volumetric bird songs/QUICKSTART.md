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
- `examples/volumetric_static.html` - **NEW! CT-scan-like solid 3D surface** (RECOMMENDED!)
- `examples/volumetric_animated.html` - **NEW! Animated with SYNCHRONIZED AUDIO** 🔊 (BEST!)
- `examples/polar_animation.html` - Animated circular visualization
- `examples/diamond_3d_mirror.html` - 3D diamond with mirrored features
- `examples/diamond_3d_multi.html` - 3D diamond with four features
- `examples/comparison_dashboard.html` - Complete analysis dashboard

## Example Output Location

After running the script, you'll find:
```
examples/
├── volumetric_static.html       ← NEW! Solid 3D surface (CT-scan-like) ⭐
├── volumetric_animated.html     ← NEW! Animated + SYNCHRONIZED AUDIO 🔊⭐⭐
├── polar_animation.html          ← Interactive circular animation
├── diamond_3d_mirror.html        ← 3D diamond (symmetric)
├── diamond_3d_multi.html         ← 3D diamond (asymmetric)
└── comparison_dashboard.html     ← Full analysis dashboard
```

## Need Help?

```bash
python volumetric_bird_songs.py --help
```

See `README.md` for detailed documentation.

## What Do the Visualizations Show?

### Volumetric Surfaces (NEW!)
- **Solid 3D mesh** = CT-scan-like volumetric surface
- **Y-axis** = Time progression
- **X-axis** = Frequency (±mirrored)
- **Z-axis** = Amplitude (±mirrored)
- **Surface color** = Amplitude intensity mapped across entire surface
- **Shape** = Overall call structure and complexity

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

1. **Start with volumetric_animated.html** - HEAR and SEE the bird song together! 🔊
2. Click "▶ Play with Audio" to experience synchronized audio+visual
3. Adjust volume slider as needed
4. Watch how the 3D surface grows as you hear the bird sing
5. Rotate while playing (click and drag) to see different perspectives
6. **Try volumetric_static.html** for the complete CT-scan-like 3D view
7. The volumetric surfaces show true solid volumes, not just scattered points
8. Listen for pitch changes and watch the surface respond in real-time!

Happy visualizing! 🐦🎵
