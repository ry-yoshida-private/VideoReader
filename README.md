# VideoReader

`VideoReader` is a Python class designed for efficient video frame extraction and iteration by opencv. 
It provides functionalities to read video frames sequentially with a specified frequency and extract individual frames by their number.

## Installation

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.video_reader.video_reader import VideoReader
import cv2

# Initialize VideoReader with default values (start=0, freq=1)
video_path = "your_video.mp4" # Replace with your video file path
reader = VideoReader(video_path)

print(f"Total frames: {len(reader)}")

# Iterate through frames
for i, frame in enumerate(reader):
    if i == 5: # Extract 6th frame (0-indexed)
        cv2.imwrite("frame_0005.jpg", frame)
    if i >= 10:
        break

# Extract a specific frame
frame_100 = reader.extract_frame(100)
if frame_100 is not None:
    cv2.imwrite("frame_0100.jpg", frame_100)

# Using with statement
with VideoReader(video_path, iter_start_frame=10, freq=5) as custom_reader:
    for i, frame in enumerate(custom_reader):
        if i == 2:
            cv2.imwrite("custom_frame_0002.jpg", frame)
        if i >= 5:
            break
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
