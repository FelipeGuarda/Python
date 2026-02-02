#!/usr/bin/env python3
"""
Volumetric Bird Song Visualization
Creates interactive visualizations of bird songs using acoustic features.

Primary: Polar/circular animated plot with frequency oscillation, amplitude size, and color brightness
Alternative: 3D diamond visualizations with time on Y-axis
"""

import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class BirdSongVisualizer:
    """Main class for bird song visualization"""
    
    def __init__(self, audio_path, sr=22050, hop_length=512):
        """
        Initialize visualizer with audio file.
        
        Parameters:
        -----------
        audio_path : str
            Path to audio file
        sr : int
            Sample rate (default 22050 Hz)
        hop_length : int
            Number of samples between frames (default 512, ~23ms at 22050Hz)
        """
        self.audio_path = audio_path
        self.sr = sr
        self.hop_length = hop_length
        self.audio = None
        self.duration = None
        self.features = {}
        
    def load_audio(self):
        """Load and preprocess audio file"""
        print(f"Loading audio: {self.audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Sample rate: {self.sr} Hz")
        return self
        
    def extract_features(self, smooth_sigma=1.0):
        """
        Extract acoustic features from audio.
        
        Parameters:
        -----------
        smooth_sigma : float
            Gaussian smoothing parameter (default 1.0)
        """
        print("Extracting acoustic features...")
        
        # Extract spectrogram for frequency analysis
        S = np.abs(librosa.stft(self.audio, hop_length=self.hop_length))
        
        # 1. Dominant Frequency using pitch tracking
        pitches, magnitudes = librosa.piptrack(
            y=self.audio, 
            sr=self.sr, 
            hop_length=self.hop_length,
            fmin=500,  # Minimum frequency for birds
            fmax=8000  # Maximum frequency for most bird songs
        )
        
        # Get dominant frequency per frame
        frequency = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                frequency.append(pitch)
            else:
                # If no pitch detected, use spectral centroid as fallback
                frequency.append(0)
        
        frequency = np.array(frequency)
        # Fill zeros with interpolation
        nonzero_indices = np.nonzero(frequency)[0]
        if len(nonzero_indices) > 1:
            frequency = np.interp(
                np.arange(len(frequency)), 
                nonzero_indices, 
                frequency[nonzero_indices]
            )
        
        # 2. Amplitude (RMS energy)
        amplitude = librosa.feature.rms(
            y=self.audio, 
            hop_length=self.hop_length
        )[0]
        
        # 3. Spectral Centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=self.audio, 
            sr=self.sr, 
            hop_length=self.hop_length
        )[0]
        
        # 4. Spectral Bandwidth (frequency spread)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=self.audio, 
            sr=self.sr, 
            hop_length=self.hop_length
        )[0]
        
        # Time array
        times = librosa.frames_to_time(
            np.arange(len(amplitude)), 
            sr=self.sr, 
            hop_length=self.hop_length
        )
        
        # Apply smoothing to reduce jitter
        if smooth_sigma > 0:
            frequency = gaussian_filter1d(frequency, smooth_sigma)
            amplitude = gaussian_filter1d(amplitude, smooth_sigma)
            spectral_centroid = gaussian_filter1d(spectral_centroid, smooth_sigma)
            spectral_bandwidth = gaussian_filter1d(spectral_bandwidth, smooth_sigma)
        
        # Store features
        self.features = {
            'time': times,
            'frequency': frequency,
            'amplitude': amplitude,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectrogram': S
        }
        
        print(f"Extracted {len(times)} time frames")
        print(f"Frequency range: {frequency.min():.0f} - {frequency.max():.0f} Hz")
        print(f"Amplitude range: {amplitude.min():.4f} - {amplitude.max():.4f}")
        
        return self
    
    def normalize_features(self):
        """Normalize features to 0-1 range for visualization"""
        print("Normalizing features...")
        
        def normalize(arr):
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max - arr_min > 0:
                return (arr - arr_min) / (arr_max - arr_min)
            return arr
        
        self.features['frequency_norm'] = normalize(self.features['frequency'])
        self.features['amplitude_norm'] = normalize(self.features['amplitude'])
        self.features['spectral_centroid_norm'] = normalize(self.features['spectral_centroid'])
        self.features['spectral_bandwidth_norm'] = normalize(self.features['spectral_bandwidth'])
        
        return self
    
    def _brightness_to_color(self, brightness_value):
        """
        Convert brightness value (0-1) to RGB color using Plasma colormap approximation.
        
        Parameters:
        -----------
        brightness_value : float
            Normalized brightness value (0-1)
        
        Returns:
        --------
        str : RGB color string
        """
        # Plasma colormap approximation (0=blue/purple, 1=yellow)
        b = float(brightness_value)
        r = min(255, int(255 * (0.05 + 0.95 * b)))
        g = min(255, int(255 * (0.03 + 0.5 * b)))
        b_channel = max(0, int(255 * (0.5 - 0.5 * b)))
        return f'rgb({r}, {g}, {b_channel})'
    
    def create_polar_animation(self, n_points=200, freq_scale=0.3, 
                              min_radius=0.2, max_radius=1.0, 
                              output_path='polar_animation.html'):
        """
        Create animated polar/circular visualization.
        
        Parameters:
        -----------
        n_points : int
            Number of points on circle circumference
        freq_scale : float
            Frequency oscillation scaling factor
        min_radius : float
            Minimum circle radius
        max_radius : float
            Maximum circle radius
        output_path : str
            Output HTML file path
        """
        print("Creating polar animation...")
        
        times = self.features['time']
        frequency = self.features['frequency']
        amplitude = self.features['amplitude_norm']
        brightness = self.features['spectral_centroid_norm']
        
        # Create frames for animation
        frames = []
        theta = np.linspace(0, 2 * np.pi, n_points)
        
        for i, t in enumerate(times):
            # Get current features
            freq = frequency[i]
            amp = amplitude[i]
            bright = brightness[i]
            
            # Create oscillating radius
            # Base radius scales with amplitude
            base_radius = min_radius + (max_radius - min_radius) * amp
            
            # Add frequency-dependent oscillation
            oscillation = freq_scale * np.sin(freq * theta / 1000.0)
            r = base_radius + oscillation * base_radius * 0.2
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Convert brightness to color
            line_color = self._brightness_to_color(bright)
            fill_color = f'rgba({int(255*bright)}, {int(100*(1-bright))}, {int(255*(1-bright))}, 0.3)'
            
            # Create frame
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=x, 
                    y=y, 
                    mode='lines',
                    line=dict(
                        color=line_color,
                        width=3
                    ),
                    fill='toself',
                    fillcolor=fill_color,
                    showlegend=False
                )],
                name=str(i),
                layout=go.Layout(
                    title=f"Time: {t:.2f}s | Freq: {freq:.0f}Hz | Amp: {amp:.2f} | Brightness: {bright:.2f}"
                )
            ))
        
        # Create initial frame
        i = 0
        freq = frequency[i]
        amp = amplitude[i]
        bright = brightness[i]
        base_radius = min_radius + (max_radius - min_radius) * amp
        oscillation = freq_scale * np.sin(freq * theta / 1000.0)
        r = base_radius + oscillation * base_radius * 0.2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Convert brightness to color
        line_color = self._brightness_to_color(bright)
        fill_color = f'rgba({int(255*bright)}, {int(100*(1-bright))}, {int(255*(1-bright))}, 0.3)'
        
        # Create figure
        fig = go.Figure(
            data=[go.Scatter(
                x=x, 
                y=y, 
                mode='lines',
                line=dict(
                    color=line_color,
                    width=3
                ),
                fill='toself',
                fillcolor=fill_color,
                showlegend=False
            )],
            frames=frames
        )
        
        # Update layout
        fig.update_layout(
            title=f"Bird Song Polar Visualization<br>{Path(self.audio_path).name}",
            xaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False,
                zeroline=True,
                showticklabels=False,
                title=""
            ),
            yaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False,
                zeroline=True,
                showticklabels=False,
                title="",
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='rgb(10, 10, 10)',
            paper_bgcolor='rgb(20, 20, 20)',
            font=dict(color='white'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'y': 0
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f"{times[int(f.name)]:.2f}s",
                        'method': 'animate'
                    }
                    for f in frames[::max(1, len(frames)//50)]  # Show max 50 slider labels
                ],
                'x': 0.1,
                'len': 0.9,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top',
                'pad': {'b': 10, 't': 50},
                'currentvalue': {
                    'visible': True,
                    'prefix': 'Time: ',
                    'xanchor': 'right',
                    'suffix': 's'
                }
            }]
        )
        
        # Save
        fig.write_html(output_path)
        print(f"Saved polar animation to: {output_path}")
        return fig
    
    def create_diamond_3d_mirrored(self, scale_factor=1.0, output_path='diamond_3d_mirror.html'):
        """
        Create 3D diamond visualization with mirrored features.
        
        Parameters:
        -----------
        scale_factor : float
            Scaling factor for visualization
        output_path : str
            Output HTML file path
        """
        print("Creating 3D diamond (mirrored) visualization...")
        
        times = self.features['time']
        frequency = self.features['frequency_norm'] * scale_factor
        amplitude = self.features['amplitude_norm'] * scale_factor
        brightness = self.features['spectral_centroid_norm']
        
        # Create vertices for diamond shape at each time point
        vertices_x = []
        vertices_y = []
        vertices_z = []
        colors = []
        
        for i, t in enumerate(times):
            # Create 4 vertices forming diamond at time t
            # +X and -X: frequency (mirrored)
            # +Z and -Z: amplitude (mirrored)
            vertices_x.extend([frequency[i], -frequency[i], 0, 0])
            vertices_y.extend([t, t, t, t])
            vertices_z.extend([0, 0, amplitude[i], -amplitude[i]])
            colors.extend([brightness[i]] * 4)
        
        # Create figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Brightness")
                ),
                name='Diamond Vertices'
            )
        ])
        
        # Add connecting lines to show diamond structure (sample every N points)
        sample_rate = max(1, len(times) // 50)
        for i in range(0, len(times), sample_rate):
            idx = i * 4
            # Connect the 4 vertices in a diamond shape
            diamond_x = [vertices_x[idx], vertices_x[idx+2], vertices_x[idx+1], 
                        vertices_x[idx+3], vertices_x[idx]]
            diamond_y = [vertices_y[idx], vertices_y[idx+2], vertices_y[idx+1], 
                        vertices_y[idx+3], vertices_y[idx]]
            diamond_z = [vertices_z[idx], vertices_z[idx+2], vertices_z[idx+1], 
                        vertices_z[idx+3], vertices_z[idx]]
            
            fig.add_trace(go.Scatter3d(
                x=diamond_x,
                y=diamond_y,
                z=diamond_z,
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"3D Diamond Visualization (Mirrored)<br>{Path(self.audio_path).name}",
            scene=dict(
                xaxis_title="← Frequency (Hz) →",
                yaxis_title="Time (s)",
                zaxis_title="← Amplitude →",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                bgcolor='rgb(10, 10, 10)'
            ),
            paper_bgcolor='rgb(20, 20, 20)',
            font=dict(color='white')
        )
        
        fig.write_html(output_path)
        print(f"Saved 3D diamond (mirrored) to: {output_path}")
        return fig
    
    def create_diamond_3d_multi(self, scale_factor=1.0, output_path='diamond_3d_multi.html'):
        """
        Create 3D diamond visualization with four independent features.
        
        Parameters:
        -----------
        scale_factor : float
            Scaling factor for visualization
        output_path : str
            Output HTML file path
        """
        print("Creating 3D diamond (multi-feature) visualization...")
        
        times = self.features['time']
        frequency = self.features['frequency_norm'] * scale_factor
        centroid = self.features['spectral_centroid_norm'] * scale_factor
        amplitude = self.features['amplitude_norm'] * scale_factor
        bandwidth = self.features['spectral_bandwidth_norm'] * scale_factor
        
        # Create vertices for diamond shape at each time point
        vertices_x = []
        vertices_y = []
        vertices_z = []
        colors = []
        
        for i, t in enumerate(times):
            # Create 4 vertices forming asymmetric diamond at time t
            # +X: frequency, -X: spectral centroid
            # +Z: amplitude, -Z: spectral bandwidth
            vertices_x.extend([frequency[i], -centroid[i], 0, 0])
            vertices_y.extend([t, t, t, t])
            vertices_z.extend([0, 0, amplitude[i], -bandwidth[i]])
            # Color by time
            colors.extend([t] * 4)
        
        # Create figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,
                    colorscale='Jet',
                    showscale=True,
                    colorbar=dict(title="Time (s)")
                ),
                name='Diamond Vertices'
            )
        ])
        
        # Add connecting lines
        sample_rate = max(1, len(times) // 50)
        for i in range(0, len(times), sample_rate):
            idx = i * 4
            diamond_x = [vertices_x[idx], vertices_x[idx+2], vertices_x[idx+1], 
                        vertices_x[idx+3], vertices_x[idx]]
            diamond_y = [vertices_y[idx], vertices_y[idx+2], vertices_y[idx+1], 
                        vertices_y[idx+3], vertices_y[idx]]
            diamond_z = [vertices_z[idx], vertices_z[idx+2], vertices_z[idx+1], 
                        vertices_z[idx+3], vertices_z[idx]]
            
            fig.add_trace(go.Scatter3d(
                x=diamond_x,
                y=diamond_y,
                z=diamond_z,
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"3D Diamond Visualization (Four Features)<br>{Path(self.audio_path).name}",
            scene=dict(
                xaxis_title="Frequency (+) / Centroid (-)",
                yaxis_title="Time (s)",
                zaxis_title="Amplitude (+) / Bandwidth (-)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                bgcolor='rgb(10, 10, 10)'
            ),
            paper_bgcolor='rgb(20, 20, 20)',
            font=dict(color='white')
        )
        
        fig.write_html(output_path)
        print(f"Saved 3D diamond (multi-feature) to: {output_path}")
        return fig
    
    def create_comparison_dashboard(self, output_path='comparison_dashboard.html'):
        """
        Create a comprehensive comparison dashboard with all visualizations.
        
        Parameters:
        -----------
        output_path : str
            Output HTML file path
        """
        print("Creating comparison dashboard...")
        
        times = self.features['time']
        frequency = self.features['frequency']
        amplitude = self.features['amplitude']
        centroid = self.features['spectral_centroid']
        bandwidth = self.features['spectral_bandwidth']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Frequency Over Time',
                'Amplitude Over Time',
                'Spectral Centroid (Brightness)',
                'Spectral Bandwidth',
                'Spectrogram',
                'Feature Summary'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Frequency
        fig.add_trace(
            go.Scatter(x=times, y=frequency, mode='lines', 
                      line=dict(color='cyan'), name='Frequency'),
            row=1, col=1
        )
        
        # Amplitude
        fig.add_trace(
            go.Scatter(x=times, y=amplitude, mode='lines', 
                      line=dict(color='magenta'), name='Amplitude'),
            row=1, col=2
        )
        
        # Spectral Centroid
        fig.add_trace(
            go.Scatter(x=times, y=centroid, mode='lines', 
                      line=dict(color='yellow'), name='Centroid'),
            row=2, col=1
        )
        
        # Spectral Bandwidth
        fig.add_trace(
            go.Scatter(x=times, y=bandwidth, mode='lines', 
                      line=dict(color='lime'), name='Bandwidth'),
            row=2, col=2
        )
        
        # Spectrogram
        S_db = librosa.amplitude_to_db(self.features['spectrogram'], ref=np.max)
        fig.add_trace(
            go.Heatmap(
                z=S_db,
                x=times,
                y=librosa.fft_frequencies(sr=self.sr),
                colorscale='Hot',
                showscale=False
            ),
            row=3, col=1
        )
        
        # Summary table
        summary_data = {
            'Feature': ['Frequency', 'Amplitude', 'Centroid', 'Bandwidth', 'Duration'],
            'Min': [
                f"{frequency.min():.1f} Hz",
                f"{amplitude.min():.4f}",
                f"{centroid.min():.1f} Hz",
                f"{bandwidth.min():.1f} Hz",
                ""
            ],
            'Max': [
                f"{frequency.max():.1f} Hz",
                f"{amplitude.max():.4f}",
                f"{centroid.max():.1f} Hz",
                f"{bandwidth.max():.1f} Hz",
                ""
            ],
            'Mean': [
                f"{frequency.mean():.1f} Hz",
                f"{amplitude.mean():.4f}",
                f"{centroid.mean():.1f} Hz",
                f"{bandwidth.mean():.1f} Hz",
                f"{self.duration:.2f} s"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(summary_data.keys()), 
                           fill_color='paleturquoise', align='left'),
                cells=dict(values=list(summary_data.values()), 
                          fill_color='lavender', align='left')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=2)
        fig.update_yaxes(title_text="Centroid (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Bandwidth (Hz)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
        
        fig.update_layout(
            title=f"Bird Song Analysis Dashboard<br>{Path(self.audio_path).name}",
            showlegend=False,
            height=1200,
            paper_bgcolor='rgb(240, 240, 240)'
        )
        
        fig.write_html(output_path)
        print(f"Saved comparison dashboard to: {output_path}")
        return fig
    
    def create_all_visualizations(self, output_dir='examples'):
        """Create all visualization types and save to output directory"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create all visualizations
        self.create_polar_animation(
            output_path=os.path.join(output_dir, 'polar_animation.html')
        )
        self.create_diamond_3d_mirrored(
            output_path=os.path.join(output_dir, 'diamond_3d_mirror.html')
        )
        self.create_diamond_3d_multi(
            output_path=os.path.join(output_dir, 'diamond_3d_multi.html')
        )
        self.create_comparison_dashboard(
            output_path=os.path.join(output_dir, 'comparison_dashboard.html')
        )
        
        print(f"\n{'='*60}")
        print(f"All visualizations saved to: {output_dir}/")
        print(f"{'='*60}")
        return self


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description='Create interactive visualizations of bird songs'
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file (WAV, MP3, etc.)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='examples',
        help='Output directory for HTML files (default: examples)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Sample rate in Hz (default: 22050)'
    )
    parser.add_argument(
        '--hop-length',
        type=int,
        default=512,
        help='Hop length for feature extraction (default: 512)'
    )
    parser.add_argument(
        '--viz-type',
        type=str,
        choices=['polar', 'diamond-mirror', 'diamond-multi', 'dashboard', 'all'],
        default='all',
        help='Type of visualization to create (default: all)'
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    # Create visualizer
    viz = BirdSongVisualizer(
        audio_path=args.audio_file,
        sr=args.sr,
        hop_length=args.hop_length
    )
    
    # Process audio
    viz.load_audio()
    viz.extract_features()
    viz.normalize_features()
    
    # Create visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.viz_type == 'all':
        viz.create_all_visualizations(output_dir=args.output_dir)
    elif args.viz_type == 'polar':
        viz.create_polar_animation(
            output_path=os.path.join(args.output_dir, 'polar_animation.html')
        )
    elif args.viz_type == 'diamond-mirror':
        viz.create_diamond_3d_mirrored(
            output_path=os.path.join(args.output_dir, 'diamond_3d_mirror.html')
        )
    elif args.viz_type == 'diamond-multi':
        viz.create_diamond_3d_multi(
            output_path=os.path.join(args.output_dir, 'diamond_3d_multi.html')
        )
    elif args.viz_type == 'dashboard':
        viz.create_comparison_dashboard(
            output_path=os.path.join(args.output_dir, 'comparison_dashboard.html')
        )
    
    print("\n✓ Done! Open the HTML files in your browser to view the visualizations.")


if __name__ == '__main__':
    main()
