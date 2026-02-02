# Volumetric Bird Song Visualization

Create stunning interactive visualizations of bird songs using acoustic feature extraction and multiple visualization techniques.

## Overview

This project provides tools to analyze and visualize bird songs in multiple ways:

1. **Polar/Circular Animation** (Primary) - An animated circular visualization where:
   - **Circumference oscillates** with frequency changes
   - **Circle size** changes with amplitude/volume
   - **Color** represents spectral centroid (brightness/timbral quality)

2. **3D Diamond Visualizations** (Alternative) - Volumetric 3D plots with:
   - Time on Y-axis
   - Acoustic features on X and Z axes
   - Two variants: mirrored (symmetric) and multi-feature (asymmetric)

3. **Comparison Dashboard** - Comprehensive view with time-series plots, spectrogram, and feature statistics

## Installation

### 1. Create Conda Environment

```bash
cd "Volumetric bird songs"
conda env create -f environment.yml
conda activate bird_viz
```

### 2. Verify Installation

```bash
python volumetric_bird_songs.py --help
```

## Usage

### Basic Usage

Place your bird song audio files in the `sample_audio/` directory, then:

```bash
# Create all visualizations
python volumetric_bird_songs.py path/to/your/birdsong.wav

# Or specify output directory
python volumetric_bird_songs.py path/to/your/birdsong.wav --output-dir my_output
```

### Create Specific Visualizations

```bash
# Only polar animation
python volumetric_bird_songs.py birdsong.wav --viz-type polar

# Only volumetric surface (static) - NEW! CT-scan-like solid 3D surface
python volumetric_bird_songs.py birdsong.wav --viz-type volumetric-static

# Only volumetric surface (animated) - NEW! Progressive build animation
python volumetric_bird_songs.py birdsong.wav --viz-type volumetric-animated

# Both volumetric surfaces - NEW!
python volumetric_bird_songs.py birdsong.wav --viz-type volumetric

# Only 3D diamond (mirrored)
python volumetric_bird_songs.py birdsong.wav --viz-type diamond-mirror

# Only 3D diamond (multi-feature)
python volumetric_bird_songs.py birdsong.wav --viz-type diamond-multi

# Only comparison dashboard
python volumetric_bird_songs.py birdsong.wav --viz-type dashboard

# All visualizations (default)
python volumetric_bird_songs.py birdsong.wav --viz-type all
```

### Advanced Options

```bash
# Custom sample rate and hop length
python volumetric_bird_songs.py birdsong.wav --sr 44100 --hop-length 256

# Full example
python volumetric_bird_songs.py my_audio.mp3 \
    --output-dir results \
    --viz-type all \
    --sr 22050 \
    --hop-length 512
```

## Acoustic Features Explained

### 1. Frequency (Hz)
- **What it is**: The dominant pitch of the sound at each moment
- **Extraction method**: `librosa.piptrack()` - pitch tracking algorithm
- **Typical range**: 500-8000 Hz for most bird songs
- **Visual mapping**: 
  - Polar plot: Controls oscillation speed of circle edge
  - 3D diamond: Extends along +X axis (or both ±X in mirrored version)

### 2. Amplitude / Volume
- **What it is**: The loudness or energy of the sound
- **Extraction method**: `librosa.feature.rms()` - Root Mean Square energy
- **Range**: Normalized to 0-1 for visualization
- **Visual mapping**:
  - Polar plot: Controls circle size (larger = louder)
  - 3D diamond: Extends along +Z axis (or both ±Z in mirrored version)

### 3. Spectral Centroid (Brightness)
- **What it is**: The "center of mass" of the frequency spectrum - indicates brightness/sharpness
- **Extraction method**: `librosa.feature.spectral_centroid()` - weighted mean of frequencies
- **Interpretation**: 
  - High value = bright, sharp, high-frequency-rich sounds
  - Low value = dark, mellow, bass-heavy sounds
- **Visual mapping**:
  - Polar plot: Controls color (warm = bright, cool = dark)
  - 3D diamond (multi): Extends along -X axis

### 4. Spectral Bandwidth
- **What it is**: The width/spread of the frequency distribution
- **Extraction method**: `librosa.feature.spectral_bandwidth()` - standard deviation of spectrum
- **Interpretation**:
  - High value = noisy, complex, broadband sounds
  - Low value = pure, tonal, narrow sounds
- **Visual mapping**:
  - 3D diamond (multi): Extends along -Z axis

## Visualization Types

### Polar Animation (`polar_animation.html`)

**Best for**: Presentations, initial exploration, educational purposes

An animated circular plot that plays through the bird song over time. The circle:
- Vibrates/oscillates based on frequency (pitch)
- Grows and shrinks based on amplitude (volume)
- Changes color based on spectral centroid (brightness)

**Controls**:
- Play/Pause buttons
- Timeline slider to jump to specific moments
- Hover for exact values

**Interpretation tips**:
- Fast vibrations = high-pitched sounds
- Large circles = loud calls
- Warm colors (red/yellow) = bright, sharp sounds
- Cool colors (blue/purple) = mellow, dark sounds

### Volumetric Surface - Static (`volumetric_static.html`) **NEW!**

**Best for**: CT-scan-like solid 3D visualization, understanding overall structure, presentations

A true volumetric 3D surface that shows the bird song as a solid, filled object:
- Y-axis = Time (vertical progression)
- X-axis = ±Frequency (mirrored symmetrically)
- Z-axis = ±Amplitude (mirrored symmetrically)
- **Surface color** = Amplitude intensity (color-mapped across entire surface)

**Key features**:
- Solid, filled triangulated mesh (not just scattered points)
- CT-scan-like appearance with smooth surfaces
- Entire surface color-mapped to amplitude using Viridis colorscale
- Interactive 3D rotation and zoom
- Shows the complete song structure at once

**Interpretation tips**:
- Wide bulges = loud or frequency-rich sections
- Narrow sections = quieter or simpler calls
- Color gradient shows amplitude variations throughout
- Overall shape reveals call complexity and temporal patterns

### Volumetric Surface - Animated (`volumetric_animated.html`) **NEW!**

**Best for**: Watching the call evolve, understanding temporal structure, dynamic presentations

Same as the static volumetric surface, but builds progressively over time **with synchronized audio playback**:
- Starts from the beginning of the audio
- Progressively reveals/builds the 3D surface as time advances
- **Plays the actual bird song synchronized with the visualization**
- Shows how the volumetric shape grows during the bird call

**Controls**:
- **▶ Play with Audio** button - plays both animation AND audio together
- **⏸ Pause** button - pauses both simultaneously
- Volume slider for audio control
- Real-time audio time display
- Interactive 3D rotation while playing
- Automatically synchronized - no manual coordination needed

**Audio Synchronization**:
- Audio embedded directly in the HTML file
- Real-time synchronization (updates every 50ms)
- Animation frames automatically match audio timestamp
- Hear and see the bird song simultaneously

**Interpretation tips**:
- **Listen and watch together** to understand how sound maps to shape
- Hear pitch changes as the shape expands/contracts
- Notice how loud parts create larger bulges in the surface
- Pause at any point to examine specific moments
- Rotate while playing to see different perspectives
- Observe how amplitude colors change through time

**Advantages over scattered-point visualizations**:
- True volumetric appearance (like medical CT scans)
- Solid surfaces make 3D shape clear and intuitive
- Color mapping across surfaces (not just vertices)
- Professional, publication-ready appearance

### 3D Diamond - Mirrored (`diamond_3d_mirror.html`)

**Best for**: Simple 3D exploration, symmetric visualization

A 3D volumetric visualization where:
- Y-axis = Time (vertical)
- X-axis = ±Frequency (mirrored symmetrically)
- Z-axis = ±Amplitude (mirrored symmetrically)
- Color = Spectral centroid (brightness)

The resulting shape shows the entire bird song as a 3D object you can rotate and explore.

### 3D Diamond - Multi-Feature (`diamond_3d_multi.html`)

**Best for**: Research, detailed analysis, comparing multiple features

A 3D volumetric visualization with four independent features:
- Y-axis = Time (vertical)
- +X axis = Frequency
- -X axis = Spectral Centroid
- +Z axis = Amplitude
- -Z axis = Spectral Bandwidth
- Color = Time progression

This creates an asymmetric shape that captures more acoustic complexity.

### Comparison Dashboard (`comparison_dashboard.html`)

**Best for**: Comprehensive analysis, understanding all features simultaneously

A multi-panel dashboard showing:
1. Frequency time series
2. Amplitude time series
3. Spectral centroid time series
4. Spectral bandwidth time series
5. Spectrogram (frequency vs time heatmap)
6. Statistical summary table

## Technical Details

### Audio Processing Pipeline

1. **Load Audio**: Load audio file at specified sample rate (default 22050 Hz)
2. **Feature Extraction**: Extract features using sliding windows
   - Window size: determined by hop_length (default 512 samples ≈ 23ms)
   - Features computed per window using librosa
3. **Smoothing**: Apply Gaussian smoothing to reduce jitter (sigma=1.0)
4. **Normalization**: Normalize features to 0-1 range for visualization
5. **Visualization**: Generate interactive HTML plots with Plotly

### Parameters You Can Adjust

In the script or via CLI:
- `sr` (sample rate): Higher = more detail but slower (default: 22050 Hz)
- `hop_length`: Smaller = smoother animation but more data (default: 512)
- `smooth_sigma`: Gaussian smoothing strength (default: 1.0)

In polar animation method:
- `n_points`: Number of points on circle (default: 200)
- `freq_scale`: Frequency oscillation strength (default: 0.3)
- `min_radius`/`max_radius`: Circle size range (default: 0.2-1.0)

### Supported Audio Formats

Thanks to librosa and soundfile, you can use:
- WAV (uncompressed)
- MP3
- FLAC
- OGG
- M4A
- And more!

## Tips and Recommendations

### For Best Results

1. **Audio Quality**: Use clean recordings with minimal background noise
2. **Duration**: Works best with 1-30 second clips (longer audio = more data = slower processing)
3. **Preprocessing**: Consider noise reduction if your audio has significant background noise

### Choosing Visualization Type

| Goal | Recommended Visualization |
|------|--------------------------|
| Presentation to non-technical audience | Polar Animation or Volumetric Animated |
| Quick exploration | Polar Animation or Volumetric Static |
| Understanding overall call structure | Volumetric Static |
| Watching call evolution over time | Volumetric Animated |
| Understanding pitch/loudness patterns | Dashboard |
| Research paper / Publication | Volumetric Static or 3D Diamond Multi-Feature |
| Comparing multiple bird species | Polar Animation (create one per species) |
| Teaching acoustics | Dashboard |
| CT-scan-like 3D visualization | Volumetric Static or Volumetric Animated |

### Interpretation Guide

**Polar Animation Patterns**:
- Steady, regular oscillation = sustained note
- Erratic oscillation = frequency modulation (trills)
- Pulsing size = amplitude modulation
- Color shifts = timbre changes

**3D Diamond Patterns**:
- Narrow diamond = pure, tonal call
- Wide diamond = loud or frequency-rich call
- Twisted shape = complex call with varying features
- Asymmetric shape (multi-feature) = acoustically diverse

## Troubleshooting

### Common Issues

**"No module named 'librosa'"**
- Solution: Make sure conda environment is activated: `conda activate bird_viz`

**"Audio file not found"**
- Solution: Check file path, use absolute path if needed

**Visualization is too fast/slow**
- Solution: Adjust `hop_length` (smaller = more frames = slower if playing)

**Circle barely oscillates in polar plot**
- Solution: Increase `freq_scale` parameter in the code (default 0.3, try 0.5-1.0)

**Python script crashes with large audio files**
- Solution: Split audio into smaller chunks or increase available RAM

## Examples

### Example Workflow

```bash
# 1. Activate environment
conda activate bird_viz

# 2. Place your audio file
cp ~/recordings/robin_song.wav sample_audio/

# 3. Create all visualizations
python volumetric_bird_songs.py sample_audio/robin_song.wav

# 4. Open results (generated in examples/ directory)
# Open examples/polar_animation.html in your browser
```

### Python API Usage

You can also use the visualizer as a Python class:

```python
from volumetric_bird_songs import BirdSongVisualizer

# Create visualizer
viz = BirdSongVisualizer('path/to/audio.wav', sr=22050, hop_length=512)

# Process audio
viz.load_audio()
viz.extract_features()
viz.normalize_features()

# Create specific visualization
viz.create_polar_animation(output_path='my_polar.html')

# Or create all visualizations
viz.create_all_visualizations(output_dir='my_results')

# Access extracted features
print(viz.features['frequency'])  # Array of frequency values
print(viz.features['amplitude'])   # Array of amplitude values
```

## Project Structure

```
Volumetric bird songs/
├── environment.yml              # Conda environment specification
├── volumetric_bird_songs.py     # Main script
├── README.md                    # This file
├── examples/                    # Output visualizations (created after first run)
│   ├── polar_animation.html
│   ├── diamond_3d_mirror.html
│   ├── diamond_3d_multi.html
│   └── comparison_dashboard.html
└── sample_audio/                # Place your audio files here
    └── .gitkeep
```

## Credits and References

**Libraries Used**:
- [librosa](https://librosa.org/) - Audio analysis
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing

**Acoustic Feature References**:
- Spectral features: [LibROSA Feature Documentation](https://librosa.org/doc/latest/feature.html)
- Bird song analysis: Sound Analysis Pro (SAP) methodology

## License

This project is provided as-is for research and educational purposes.

## Contributing

Feel free to extend this project by:
- Adding new visualization types
- Implementing additional acoustic features
- Creating comparative analysis tools
- Adding audio preprocessing options

## Questions?

For questions about acoustic features or visualization interpretation, consult:
- [LibROSA documentation](https://librosa.org/doc/latest/index.html)
- [Plotly documentation](https://plotly.com/python/)
- Bird song analysis literature

---

**Enjoy visualizing bird songs! 🐦🎵📊**
