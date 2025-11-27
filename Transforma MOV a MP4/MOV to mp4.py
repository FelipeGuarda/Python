from moviepy.editor import VideoFileClip

# Load your .AVI file (replace 'input.avi' with actual file name)
clip = VideoFileClip("monito_del_monte.AVI")
# Write to .mp4, using video codec 'libx264' and audio codec 'aac'
clip.write_videofile("output_monito.mp4", codec="libx264", audio_codec="aac")
