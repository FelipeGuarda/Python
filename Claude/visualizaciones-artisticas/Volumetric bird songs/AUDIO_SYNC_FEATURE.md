# Audio Synchronization Feature

## Overview

The volumetric animated visualization now includes **real-time synchronized audio playback**, allowing you to hear and see the bird song simultaneously!

## What Was Added

### 1. Audio Embedding
- The actual bird song audio is embedded directly into the HTML file
- Converted to WAV format and base64-encoded
- No external audio files needed - everything is in one HTML file

### 2. Synchronized Playback
- Audio and animation are perfectly synchronized
- Updates every 50ms for smooth coordination
- Animation frames automatically match the audio timestamp

### 3. Custom Audio Controls
- **"▶ Play with Audio"** button - starts both audio and animation together
- **"⏸ Pause"** button - pauses both simultaneously
- **Volume slider** - adjust audio volume (0-100%)
- **Real-time display** - shows current audio time and total duration
- **No manual coordination needed** - everything is automatic!

## User Interface

The audio controls appear as a fixed overlay at the bottom of the page:

```
┌─────────────────────────────────────────────────┐
│  ▶ Play with Audio                             │
│  Audio Time: 2.45s / 5.01s                     │
│  Volume: ━━━━━━━━◉━━ 70%                       │
└─────────────────────────────────────────────────┘
```

### Visual Design
- Dark semi-transparent background for visibility
- Green "Play" button, red "Pause" button
- Monospace font for time display
- Positioned at bottom center (doesn't block visualization)

## How It Works

### Technical Implementation

1. **Audio Processing**
   - Original audio loaded by librosa
   - Converted to WAV format in memory using soundfile
   - Base64-encoded for embedding in HTML
   - Creates data URI: `data:audio/wav;base64,...`

2. **Synchronization Logic**
   ```javascript
   // Every 50ms:
   - Get current audio playback time
   - Find closest animation frame for that time
   - Update visualization to match
   - Display current time
   ```

3. **Frame Timing**
   - Each animation frame knows its timestamp
   - JavaScript finds the best matching frame
   - Uses Plotly's animate() function to jump to that frame

4. **Auto-sync Updates**
   - setInterval runs every 50ms during playback
   - Continuously keeps animation aligned with audio
   - Clears interval when paused or ended

## Usage

### Basic Usage
```bash
# Create animated visualization with audio
python volumetric_bird_songs.py "your_audio.wav" --viz-type volumetric-animated

# Or create all visualizations (includes audio-enabled animation)
python volumetric_bird_songs.py "your_audio.wav" --viz-type all
```

### Opening the File
1. Open `examples/volumetric_animated.html` in your browser
2. Click "▶ Play with Audio" button
3. Hear the bird song while watching the 3D surface build
4. Adjust volume as needed
5. Pause/restart anytime

### Controls During Playback
- **Click and drag** to rotate the 3D visualization while it plays
- **Scroll** to zoom in/out
- **Volume slider** to adjust audio level
- **Pause button** to stop and examine a specific moment
- **Play again** to restart from current position

## Benefits

### For Analysis
- **Correlate sound with shape**: See exactly how pitch and volume affect the visualization
- **Identify patterns**: Match acoustic features to visual structures
- **Teaching tool**: Perfect for explaining bioacoustics concepts
- **Engagement**: More engaging than silent animations

### For Understanding
- **Pitch changes**: Hear high notes, see the shape narrow/widen
- **Volume variations**: Hear loud parts, see the surface expand
- **Call structure**: Understand temporal patterns by hearing+seeing
- **Species comparison**: Compare calls both aurally and visually

## Example Use Cases

1. **Bird Song Analysis**
   - Play the animation and listen for pitch modulations
   - Watch how the volumetric surface responds to frequency changes
   - Identify call segments by their visual+audio signature

2. **Educational Demonstrations**
   - Show students the relationship between sound and visualization
   - Explain acoustic features with real-time examples
   - Compare different species' calls interactively

3. **Research Presentations**
   - Present findings with synchronized audio-visual evidence
   - Demonstrate call complexity dynamically
   - Engage audience with multi-sensory experience

4. **Species Documentation**
   - Create comprehensive records with both modalities
   - Archive calls with corresponding visualizations
   - Share via single HTML file (no dependencies)

## Technical Details

### File Size Impact
- Audio embedded adds ~400KB for a 5-second WAV file
- Compression maintains reasonable file sizes
- Example: 5-second bird call = ~1.9MB total HTML file
- Trade-off: Larger file but self-contained and no external dependencies

### Browser Compatibility
- Works in all modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- Audio playback may require user interaction (browser security)
- HTML5 Audio API provides cross-platform compatibility

### Performance
- Lightweight synchronization (50ms updates)
- No impact on 3D rendering performance
- Audio decoding handled by browser
- Smooth playback on modern hardware

## Limitations and Notes

1. **Browser Auto-play Policies**
   - Some browsers require user interaction before audio playback
   - If audio doesn't start, click anywhere on the page first

2. **File Size**
   - Longer audio files create larger HTML files
   - Keep clips under 30 seconds for optimal file size

3. **Audio Format**
   - Currently converts to WAV for compatibility
   - MP3/other formats converted on embedding

4. **Synchronization Accuracy**
   - 50ms update interval = excellent sync
   - Typically imperceptible lag
   - Frame-based animation limits (discrete frames, not continuous)

## Code Implementation

### New Methods Added

```python
_audio_to_base64()
```
- Converts loaded audio to base64-encoded WAV data URI
- Uses soundfile to write WAV to memory buffer
- Returns string ready for HTML embedding

### Modified Methods

```python
create_volumetric_surface_animated()
```
- Now embeds audio in the output HTML
- Adds custom JavaScript for synchronization
- Creates audio player UI overlay
- Implements play/pause/volume controls

## Future Enhancements (Potential)

- [ ] Adjustable playback speed (faster/slower)
- [ ] Loop mode for continuous playback
- [ ] Audio waveform display alongside visualization
- [ ] Export as video file (with audio track)
- [ ] Multiple audio tracks (original + filtered versions)
- [ ] Spectral visualization alongside time-domain

## Credits

- Audio embedding: Python soundfile + base64
- Synchronization: JavaScript + HTML5 Audio API
- Visualization: Plotly.js 3D mesh animations
- UI: Custom HTML/CSS/JavaScript overlay

---

**This feature transforms the volumetric visualization from a visual-only tool to a true multi-sensory experience!** 🎵📊🔊
