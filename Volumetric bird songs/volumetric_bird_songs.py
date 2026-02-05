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
import base64
import io
import soundfile as sf
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
    
    def _audio_to_base64(self):
        """
        Convert audio to base64-encoded WAV data URI for embedding in HTML.
        
        Returns:
        --------
        str : Base64-encoded data URI
        """
        # Create WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, self.audio, self.sr, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        data_uri = f'data:audio/wav;base64,{audio_base64}'
        
        return data_uri
    
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
    
    def create_volumetric_surface_static(self, scale_factor=1.0, opacity=0.9, 
                                         colorscale='Viridis', 
                                         output_path='volumetric_static.html'):
        """
        Create static volumetric surface visualization using Mesh3d.
        Creates a solid, CT-scan-like 3D surface by connecting diamond cross-sections.
        
        Parameters:
        -----------
        scale_factor : float
            Scaling factor for visualization (default 1.0)
        opacity : float
            Surface opacity, 0-1 (default 0.9 for solid appearance)
        colorscale : str
            Plotly colorscale name (default 'Viridis')
        output_path : str
            Output HTML file path
        """
        print("Creating volumetric surface (static)...")
        
        times = self.features['time']
        frequency = self.features['frequency_norm'] * scale_factor
        amplitude = self.features['amplitude_norm'] * scale_factor
        amplitude_raw = self.features['amplitude']  # For color mapping
        
        # Build vertex arrays
        vertices_x = []
        vertices_y = []
        vertices_z = []
        colors = []
        
        for i, t in enumerate(times):
            freq = frequency[i]
            amp = amplitude[i]
            amp_color = amplitude_raw[i]
            
            # 4 vertices per time slice forming diamond cross-section
            # Vertex order: [+X, -X, +Z, -Z]
            vertices_x.extend([freq, -freq, 0, 0])
            vertices_y.extend([t, t, t, t])
            vertices_z.extend([0, 0, amp, -amp])
            colors.extend([amp_color] * 4)  # Color by amplitude
        
        # Build triangulation indices
        # Connect consecutive diamond slices with triangular faces
        i_indices = []
        j_indices = []
        k_indices = []
        
        for t_idx in range(len(times) - 1):
            base = t_idx * 4  # Current slice base index
            next_base = (t_idx + 1) * 4  # Next slice base index
            
            # Define 8 triangles connecting the two diamond slices
            # Each face uses 2 triangles to form a quad
            triangles = [
                # Face 1: Connect vertices 0 and 2 (front right quadrant)
                [base+0, next_base+0, base+2],
                [next_base+0, next_base+2, base+2],
                # Face 2: Connect vertices 2 and 1 (front left quadrant)
                [base+2, next_base+2, base+1],
                [next_base+2, next_base+1, base+1],
                # Face 3: Connect vertices 1 and 3 (back left quadrant)
                [base+1, next_base+1, base+3],
                [next_base+1, next_base+3, base+3],
                # Face 4: Connect vertices 3 and 0 (back right quadrant)
                [base+3, next_base+3, base+0],
                [next_base+3, next_base+0, base+0],
            ]
            
            for tri in triangles:
                i_indices.append(tri[0])
                j_indices.append(tri[1])
                k_indices.append(tri[2])
        
        # Create Mesh3d figure
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=i_indices,
                j=j_indices,
                k=k_indices,
                intensity=colors,
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(
                    title='Amplitude',
                    thickness=20,
                    len=0.7,
                    x=1.02
                ),
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.2,
                    roughness=0.5,
                    fresnel=0.2
                ),
                flatshading=False,
                name='Volumetric Surface'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"Volumetric Surface Visualization<br>{Path(self.audio_path).name}",
            scene=dict(
                xaxis_title="← Frequency (Hz) →",
                yaxis_title="Time (s)",
                zaxis_title="← Amplitude →",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    up=dict(x=0, y=1, z=0)
                ),
                aspectmode='data',
                bgcolor='rgb(10, 10, 10)'
            ),
            paper_bgcolor='rgb(20, 20, 20)',
            font=dict(color='white'),
            width=1000,
            height=800
        )
        
        fig.write_html(output_path)
        print(f"Saved volumetric surface (static) to: {output_path}")
        return fig
    
    def create_volumetric_surface_animated(self, scale_factor=1.0, opacity=0.9,
                                          colorscale='Viridis', frame_step=5,
                                          output_path='volumetric_animated.html'):
        """
        Create animated volumetric surface that builds progressively over time.
        
        Parameters:
        -----------
        scale_factor : float
            Scaling factor for visualization (default 1.0)
        opacity : float
            Surface opacity, 0-1 (default 0.9)
        colorscale : str
            Plotly colorscale name (default 'Viridis')
        frame_step : int
            Number of time slices to add per frame (default 5)
        output_path : str
            Output HTML file path
        """
        print("Creating volumetric surface (animated)...")
        
        times = self.features['time']
        frequency = self.features['frequency_norm'] * scale_factor
        amplitude = self.features['amplitude_norm'] * scale_factor
        amplitude_raw = self.features['amplitude']
        
        # Build complete vertex arrays
        vertices_x = []
        vertices_y = []
        vertices_z = []
        colors = []
        
        for i, t in enumerate(times):
            freq = frequency[i]
            amp = amplitude[i]
            amp_color = amplitude_raw[i]
            
            vertices_x.extend([freq, -freq, 0, 0])
            vertices_y.extend([t, t, t, t])
            vertices_z.extend([0, 0, amp, -amp])
            colors.extend([amp_color] * 4)
        
        # Calculate fixed axis ranges based on complete data extent
        # Add padding for better visualization
        padding = 0.1
        x_min, x_max = min(vertices_x), max(vertices_x)
        y_min, y_max = min(vertices_y), max(vertices_y)
        z_min, z_max = min(vertices_z), max(vertices_z)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        fixed_x_range = [x_min - padding * x_range, x_max + padding * x_range]
        fixed_y_range = [y_min - padding * y_range, y_max + padding * y_range]
        fixed_z_range = [z_min - padding * z_range, z_max + padding * z_range]
        
        print(f"  Fixed X range: [{fixed_x_range[0]:.2f}, {fixed_x_range[1]:.2f}]")
        print(f"  Fixed Y range: [{fixed_y_range[0]:.2f}, {fixed_y_range[1]:.2f}]")
        print(f"  Fixed Z range: [{fixed_z_range[0]:.2f}, {fixed_z_range[1]:.2f}]")
        
        # Create animation frames
        frames = []
        
        # Sample frames to avoid too many (max ~50 frames)
        frame_indices = list(range(1, len(times), max(1, frame_step)))
        if len(times) - 1 not in frame_indices:
            frame_indices.append(len(times) - 1)
        
        for end_idx in frame_indices:
            # Build triangulation for vertices from 0 to end_idx
            i_indices = []
            j_indices = []
            k_indices = []
            
            for t_idx in range(end_idx):
                base = t_idx * 4
                next_base = (t_idx + 1) * 4
                
                triangles = [
                    [base+0, next_base+0, base+2],
                    [next_base+0, next_base+2, base+2],
                    [base+2, next_base+2, base+1],
                    [next_base+2, next_base+1, base+1],
                    [base+1, next_base+1, base+3],
                    [next_base+1, next_base+3, base+3],
                    [base+3, next_base+3, base+0],
                    [next_base+3, next_base+0, base+0],
                ]
                
                for tri in triangles:
                    i_indices.append(tri[0])
                    j_indices.append(tri[1])
                    k_indices.append(tri[2])
            
            # Get vertices up to current time
            max_vert_idx = (end_idx + 1) * 4
            frame_vx = vertices_x[:max_vert_idx]
            frame_vy = vertices_y[:max_vert_idx]
            frame_vz = vertices_z[:max_vert_idx]
            frame_colors = colors[:max_vert_idx]
            
            frames.append(go.Frame(
                data=[go.Mesh3d(
                    x=frame_vx,
                    y=frame_vy,
                    z=frame_vz,
                    i=i_indices,
                    j=j_indices,
                    k=k_indices,
                    intensity=frame_colors,
                    colorscale=colorscale,
                    opacity=opacity,
                    colorbar=dict(
                        title='Amplitude',
                        thickness=20,
                        len=0.7,
                        x=1.02
                    ),
                    lighting=dict(
                        ambient=0.5,
                        diffuse=0.8,
                        specular=0.2,
                        roughness=0.5,
                        fresnel=0.2
                    ),
                    flatshading=False,
                    name='Volumetric Surface'
                )],
                name=str(end_idx),
                layout=go.Layout(
                    title=f"Time: {times[end_idx]:.2f}s"
                )
            ))
        
        # Create initial frame (first few slices)
        initial_idx = min(5, len(times) - 1)
        i_init = []
        j_init = []
        k_init = []
        
        for t_idx in range(initial_idx):
            base = t_idx * 4
            next_base = (t_idx + 1) * 4
            triangles = [
                [base+0, next_base+0, base+2],
                [next_base+0, next_base+2, base+2],
                [base+2, next_base+2, base+1],
                [next_base+2, next_base+1, base+1],
                [base+1, next_base+1, base+3],
                [next_base+1, next_base+3, base+3],
                [base+3, next_base+3, base+0],
                [next_base+3, next_base+0, base+0],
            ]
            for tri in triangles:
                i_init.append(tri[0])
                j_init.append(tri[1])
                k_init.append(tri[2])
        
        max_init = (initial_idx + 1) * 4
        
        # Create figure
        fig = go.Figure(
            data=[go.Mesh3d(
                x=vertices_x[:max_init],
                y=vertices_y[:max_init],
                z=vertices_z[:max_init],
                i=i_init,
                j=j_init,
                k=k_init,
                intensity=colors[:max_init],
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(
                    title='Amplitude',
                    thickness=20,
                    len=0.7,
                    x=1.02
                ),
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.2,
                    roughness=0.5,
                    fresnel=0.2
                ),
                flatshading=False,
                name='Volumetric Surface'
            )],
            frames=frames
        )
        
        # Fixed camera position for stable viewing during playback
        # Position: looking from front-right-above, watching time progress along Y axis
        fixed_camera = dict(
            eye=dict(x=1.5, y=-0.3, z=0.8),  # Front-right view
            up=dict(x=0, y=0, z=1),           # Z is up
            center=dict(x=0, y=0.3, z=0)      # Look slightly ahead in time
        )
        
        # Add animation controls
        fig.update_layout(
            title=f"Volumetric Bird Song<br><sub>{Path(self.audio_path).name}</sub>",
            scene=dict(
                # Fixed axis ranges to prevent jumping
                xaxis=dict(
                    title="Frequency",
                    range=fixed_x_range,
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    showticklabels=False,  # Hide tick labels for cleaner view
                    showline=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Time",
                    range=fixed_y_range,
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    showticklabels=False,  # Hide tick labels for cleaner view
                    showline=False,
                    zeroline=False,
                ),
                zaxis=dict(
                    title="Amplitude",
                    range=fixed_z_range,
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    showticklabels=False,  # Hide tick labels for cleaner view
                    showline=False,
                    zeroline=False,
                ),
                camera=fixed_camera,
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=1),  # Elongate along time axis
                bgcolor='rgb(10, 10, 10)'
            ),
            paper_bgcolor='rgb(20, 20, 20)',
            font=dict(color='white'),
            width=1000,
            height=800,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
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
                    for f in frames[::max(1, len(frames)//20)]
                ],
                'x': 0.1,
                'len': 0.85,
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
        
        # Generate base HTML from Plotly
        html_string = fig.to_html(include_plotlyjs='cdn')
        
        # Embed audio and add synchronization JavaScript
        audio_data_uri = self._audio_to_base64()
        
        # Calculate frame timings (convert to plain Python floats for JavaScript)
        frame_times = [float(times[int(f.name)]) for f in frames]
        
        # Create unified audio-visual synchronization with single Play/Pause button
        sync_script = f'''
<style>
/* Hide Plotly's default Play/Pause buttons but keep the slider for testing */
.updatemenu-container .updatemenu-button {{
    display: none !important;
}}
/* Keep slider visible for testing - drag it to verify animation works */

/* Unified control panel */
#unified-control {{
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10000;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 15px 25px;
    border-radius: 50px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: white;
    display: flex;
    align-items: center;
    gap: 15px;
}}

#unified-play-btn {{
    background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
    color: white;
    border: none;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(233,69,96,0.4);
}}

#unified-play-btn:hover {{
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(233,69,96,0.6);
}}

#unified-play-btn:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}}

#control-info {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

#control-status {{
    font-size: 14px;
    font-weight: bold;
}}

#control-time {{
    font-size: 12px;
    font-family: monospace;
    opacity: 0.8;
}}

/* Progress bar */
#progress-container {{
    width: 200px;
    height: 6px;
    background: rgba(255,255,255,0.2);
    border-radius: 3px;
    overflow: hidden;
    cursor: pointer;
}}

#progress-bar {{
    height: 100%;
    background: linear-gradient(90deg, #e94560, #0f3460);
    width: 0%;
    transition: width 0.1s linear;
}}

/* Volume control */
#volume-control {{
    display: flex;
    align-items: center;
    gap: 8px;
}}

#volume-slider {{
    width: 80px;
    height: 4px;
    -webkit-appearance: none;
    background: rgba(255,255,255,0.3);
    border-radius: 2px;
    cursor: pointer;
}}

#volume-slider::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    background: #e94560;
    border-radius: 50%;
    cursor: pointer;
}}

/* Mode toggle button */
#mode-toggle {{
    background: rgba(255,255,255,0.1);
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
    padding: 8px 12px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s ease;
    white-space: nowrap;
}}

#mode-toggle:hover {{
    background: rgba(255,255,255,0.2);
}}

#mode-toggle.explore-mode {{
    background: rgba(233,69,96,0.3);
    border-color: #e94560;
}}
</style>

<div id="unified-control">
    <button id="unified-play-btn" title="Play/Pause">▶</button>
    <div id="control-info">
        <div id="control-status">Loading audio...</div>
        <div id="control-time">0:00 / 0:00</div>
    </div>
    <div id="progress-container">
        <div id="progress-bar"></div>
    </div>
    <div id="volume-control">
        <span style="font-size: 14px;">🔊</span>
        <input type="range" id="volume-slider" min="0" max="100" value="75">
    </div>
    <button id="mode-toggle" title="Toggle between Watch Mode (fixed camera) and Explore Mode (rotate freely)">🎬 Watch</button>
</div>

<script>
// Unified audio-visual synchronization system
(function() {{
    console.log('🎵 Initializing unified audio-visual sync...');
    
    const frameTimestamps = {frame_times};
    const duration = {float(self.duration)};
    const totalFrames = frameTimestamps.length;
    
    // Fixed camera position for stable viewing during playback
    const fixedCamera = {{
        eye: {{ x: 1.5, y: -0.3, z: 0.8 }},
        up: {{ x: 0, y: 0, z: 1 }},
        center: {{ x: 0, y: 0.3, z: 0 }}
    }};
    
    console.log('📊 Total frames:', totalFrames);
    console.log('⏱️ Duration:', duration.toFixed(2), 'seconds');
    console.log('🎬 First few timestamps:', frameTimestamps.slice(0, 5));
    
    // UI Elements
    const playBtn = document.getElementById('unified-play-btn');
    const statusEl = document.getElementById('control-status');
    const timeEl = document.getElementById('control-time');
    const progressBar = document.getElementById('progress-bar');
    const progressContainer = document.getElementById('progress-container');
    const volumeSlider = document.getElementById('volume-slider');
    const modeToggle = document.getElementById('mode-toggle');
    
    // Create audio element with embedded data
    const audio = document.createElement('audio');
    audio.preload = 'auto';
    audio.volume = 0.75;
    
    // State
    let isPlaying = false;
    let isExploreMode = false;  // false = Watch Mode (fixed camera), true = Explore Mode (free rotation)
    let isLoaded = false;
    let animationFrame = null;
    let plotlyDiv = null;
    let lastFrameIdx = -1;
    
    // Check for Plotly frames after a short delay
    setTimeout(function() {{
        const div = document.querySelector('.plotly-graph-div');
        if (div && div._fullLayout && div._fullLayout._plots) {{
            console.log('✅ Plotly div found');
        }}
        if (div && div._transitionData && div._transitionData._frames) {{
            const frames = div._transitionData._frames;
            console.log('🎬 Plotly frames available:', frames.length);
            if (frames.length > 0) {{
                console.log('🎬 Frame names:', frames.slice(0, 5).map(f => f.name));
            }}
        }} else {{
            console.warn('⚠️ No Plotly frames found! Animation may not work.');
            console.log('div._transitionData:', div ? div._transitionData : 'no div');
        }}
    }}, 1000);
    
    // Format time as M:SS
    function formatTime(seconds) {{
        if (isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins + ':' + secs.toString().padStart(2, '0');
    }}
    
    function updateStatus(msg) {{
        if (statusEl) statusEl.textContent = msg;
    }}
    
    function updateTimeDisplay() {{
        if (timeEl) {{
            timeEl.textContent = formatTime(audio.currentTime) + ' / ' + formatTime(duration);
        }}
        if (progressBar) {{
            const pct = (audio.currentTime / duration) * 100;
            progressBar.style.width = pct + '%';
        }}
    }}
    
    // Find Plotly div
    function findPlotlyDiv() {{
        if (!plotlyDiv) {{
            plotlyDiv = document.querySelector('.plotly-graph-div');
        }}
        return plotlyDiv;
    }}
    
    // Binary search for closest frame
    function findClosestFrame(time) {{
        let low = 0;
        let high = frameTimestamps.length - 1;
        
        while (low < high) {{
            const mid = Math.floor((low + high) / 2);
            if (frameTimestamps[mid] < time) {{
                low = mid + 1;
            }} else {{
                high = mid;
            }}
        }}
        
        // Check if previous frame is closer
        if (low > 0 && Math.abs(frameTimestamps[low - 1] - time) < Math.abs(frameTimestamps[low] - time)) {{
            return low - 1;
        }}
        return low;
    }}
    
    // Get frames from Plotly figure for direct data access
    let plotlyFrames = null;
    
    function getPlotlyFrames() {{
        if (plotlyFrames) return plotlyFrames;
        const div = findPlotlyDiv();
        if (div && div._transitionData && div._transitionData._frames) {{
            plotlyFrames = div._transitionData._frames;
            console.log('✅ Cached', plotlyFrames.length, 'Plotly frames');
        }}
        return plotlyFrames;
    }}
    
    // Lock camera to fixed position (for Watch mode)
    function lockCamera(div) {{
        if (!div || isExploreMode) return;
        
        try {{
            Plotly.relayout(div, {{
                'scene.camera': fixedCamera
            }});
        }} catch(e) {{
            // Silently ignore camera lock errors
        }}
    }}
    
    // Main sync loop using requestAnimationFrame for smooth animation
    let cameraLockCounter = 0;
    
    function syncLoop() {{
        if (!isPlaying) return;
        
        const div = findPlotlyDiv();
        if (!div) {{
            console.warn('Plotly div not found, retrying...');
            animationFrame = requestAnimationFrame(syncLoop);
            return;
        }}
        
        const currentTime = audio.currentTime;
        const frameIdx = findClosestFrame(currentTime);
        
        // Only update if frame changed
        if (frameIdx !== lastFrameIdx) {{
            lastFrameIdx = frameIdx;
            
            // Update status to show current frame
            const modeLabel = isExploreMode ? '🔄 Explore' : '🎬 Watch';
            updateStatus(modeLabel + ': ' + frameIdx + '/' + totalFrames);
            
            // Try direct data update first (more reliable)
            const frames = getPlotlyFrames();
            if (frames && frames[frameIdx] && frames[frameIdx].data) {{
                try {{
                    // Build layout with fixed camera if in Watch mode
                    let layout = div.layout;
                    if (!isExploreMode) {{
                        layout = Object.assign({{}}, div.layout, {{
                            scene: Object.assign({{}}, div.layout.scene, {{
                                camera: fixedCamera
                            }})
                        }});
                    }}
                    
                    // Directly update the trace data from the frame
                    Plotly.react(div, frames[frameIdx].data, layout);
                }} catch(e) {{
                    console.error('React error:', e);
                    // Fallback to animate
                    try {{
                        Plotly.animate(div, [String(frameIdx)], {{
                            mode: 'immediate',
                            transition: {{ duration: 0 }},
                            frame: {{ duration: 0, redraw: true }}
                        }});
                        // Lock camera after animate
                        if (!isExploreMode) lockCamera(div);
                    }} catch(e2) {{
                        console.error('Animate fallback error:', e2);
                    }}
                }}
            }} else {{
                // Use animate as fallback
                try {{
                    Plotly.animate(div, [String(frameIdx)], {{
                        mode: 'immediate',
                        transition: {{ duration: 0 }},
                        frame: {{ duration: 0, redraw: true }}
                    }});
                    // Lock camera after animate
                    if (!isExploreMode) lockCamera(div);
                }} catch(e) {{
                    console.error('Animate error:', e);
                }}
            }}
        }} else {{
            // Even if frame didn't change, periodically lock camera in Watch mode
            cameraLockCounter++;
            if (!isExploreMode && cameraLockCounter % 10 === 0) {{
                lockCamera(div);
            }}
        }}
        
        updateTimeDisplay();
        animationFrame = requestAnimationFrame(syncLoop);
    }}
    
    // Play/Pause toggle
    function togglePlayback() {{
        if (!isLoaded) {{
            updateStatus('Still loading...');
            return;
        }}
        
        if (isPlaying) {{
            // Pause
            isPlaying = false;
            audio.pause();
            if (animationFrame) {{
                cancelAnimationFrame(animationFrame);
                animationFrame = null;
            }}
            playBtn.textContent = '▶';
            updateStatus(isExploreMode ? 'Paused (Explore Mode)' : 'Paused (Watch Mode)');
        }} else {{
            // Play
            isPlaying = true;
            playBtn.textContent = '⏸';
            cameraLockCounter = 0;
            
            // Reset camera to fixed position when starting in Watch mode
            if (!isExploreMode) {{
                const div = findPlotlyDiv();
                if (div) {{
                    lockCamera(div);
                }}
            }}
            
            updateStatus(isExploreMode ? 'Playing (Explore)' : 'Playing (Watch)');
            
            audio.play()
                .then(() => {{
                    console.log('▶️ Playback started');
                    syncLoop();
                }})
                .catch(e => {{
                    console.error('❌ Play failed:', e);
                    isPlaying = false;
                    playBtn.textContent = '▶';
                    updateStatus('Error: ' + (e.message || 'Cannot play'));
                }});
        }}
    }}
    
    // Reset to beginning
    function resetPlayback() {{
        audio.currentTime = 0;
        lastFrameIdx = -1;
        updateTimeDisplay();
        
        // Show first frame
        const div = findPlotlyDiv();
        if (div) {{
            try {{
                Plotly.animate(div, ['0'], {{
                    mode: 'immediate',
                    transition: {{ duration: 0 }},
                    frame: {{ duration: 0, redraw: false }}
                }});
            }} catch(e) {{}}
        }}
    }}
    
    // Audio events
    audio.addEventListener('loadeddata', () => {{
        isLoaded = true;
        updateStatus('Ready - Click to play');
        console.log('✅ Audio loaded');
    }});
    
    audio.addEventListener('canplaythrough', () => {{
        isLoaded = true;
        updateStatus('Ready - Click to play');
    }});
    
    audio.addEventListener('error', (e) => {{
        console.error('❌ Audio error:', audio.error);
        updateStatus('Audio error: ' + (audio.error?.message || 'Failed to load'));
    }});
    
    audio.addEventListener('ended', () => {{
        isPlaying = false;
        playBtn.textContent = '▶';
        updateStatus('Finished - Click to replay');
        if (animationFrame) {{
            cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }}
        resetPlayback();
    }});
    
    // Click handlers
    playBtn.addEventListener('click', togglePlayback);
    
    // Progress bar seek
    progressContainer.addEventListener('click', (e) => {{
        if (!isLoaded) return;
        const rect = progressContainer.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        audio.currentTime = pct * duration;
        lastFrameIdx = -1;
        updateTimeDisplay();
    }});
    
    // Volume control
    volumeSlider.addEventListener('input', (e) => {{
        audio.volume = e.target.value / 100;
    }});
    
    // Mode toggle: Watch Mode (fixed camera) vs Explore Mode (free rotation)
    function toggleMode() {{
        isExploreMode = !isExploreMode;
        
        if (isExploreMode) {{
            modeToggle.textContent = '🔄 Explore';
            modeToggle.classList.add('explore-mode');
            console.log('🔄 Switched to Explore Mode - rotate freely');
        }} else {{
            modeToggle.textContent = '🎬 Watch';
            modeToggle.classList.remove('explore-mode');
            console.log('🎬 Switched to Watch Mode - fixed camera');
            
            // Reset camera to fixed position
            const div = findPlotlyDiv();
            if (div) {{
                lockCamera(div);
            }}
        }}
        
        // Update status if not playing
        if (!isPlaying) {{
            updateStatus(isExploreMode ? 'Explore Mode - Rotate freely' : 'Watch Mode - Fixed view');
        }}
    }}
    
    modeToggle.addEventListener('click', toggleMode);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {{
        if (e.code === 'Space' && e.target === document.body) {{
            e.preventDefault();
            togglePlayback();
        }}
        // 'M' key to toggle mode
        if (e.code === 'KeyM' && e.target === document.body) {{
            e.preventDefault();
            toggleMode();
        }}
    }});
    
    // Load the audio - embed as base64 data URI
    console.log('📀 Loading audio...');
    try {{
        audio.src = '{audio_data_uri}';
    }} catch(e) {{
        console.error('❌ Failed to set audio source:', e);
        updateStatus('Failed to load audio');
    }}
    
    // Initial time display
    updateTimeDisplay();
    
    // Set initial camera to fixed position
    setTimeout(() => {{
        const div = findPlotlyDiv();
        if (div) {{
            lockCamera(div);
        }}
    }}, 500);
    
    console.log('🎵 Unified control ready');
    console.log('💡 Tip: Press M to toggle between Watch/Explore modes');
}})();
</script>
'''
        
        # Insert the synchronization script before closing body tag
        html_with_audio = html_string.replace('</body>', sync_script + '</body>')
        
        # Write modified HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_with_audio)
        
        print(f"Saved volumetric surface (animated with audio) to: {output_path}")
        return fig
    
    def create_volumetric_surfaces(self, output_dir='examples', **kwargs):
        """
        Convenience method to create both static and animated volumetric surfaces.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for HTML files
        **kwargs : dict
            Additional parameters passed to surface creation methods
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.create_volumetric_surface_static(
            output_path=os.path.join(output_dir, 'volumetric_static.html'),
            **kwargs
        )
        self.create_volumetric_surface_animated(
            output_path=os.path.join(output_dir, 'volumetric_animated.html'),
            **kwargs
        )
        
        return self
    
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
        # NEW: Volumetric CT-scan-like surfaces
        self.create_volumetric_surfaces(output_dir=output_dir)
        
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
        choices=['polar', 'diamond-mirror', 'diamond-multi', 
                'volumetric-static', 'volumetric-animated', 'volumetric',
                'dashboard', 'all'],
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
    elif args.viz_type == 'volumetric-static':
        viz.create_volumetric_surface_static(
            output_path=os.path.join(args.output_dir, 'volumetric_static.html')
        )
    elif args.viz_type == 'volumetric-animated':
        viz.create_volumetric_surface_animated(
            output_path=os.path.join(args.output_dir, 'volumetric_animated.html')
        )
    elif args.viz_type == 'volumetric':
        viz.create_volumetric_surfaces(output_dir=args.output_dir)
    elif args.viz_type == 'dashboard':
        viz.create_comparison_dashboard(
            output_path=os.path.join(args.output_dir, 'comparison_dashboard.html')
        )
    
    print("\n✓ Done! Open the HTML files in your browser to view the visualizations.")


if __name__ == '__main__':
    main()
