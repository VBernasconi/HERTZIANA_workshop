# HERTZIANA_workshop

## Living Pose Gallery
An interactive artwork discovery system that connects human body movement with visual art.

Living Pose Gallery uses real-time body pose detection to find the closest matching human pose within a collection of artworks. As you move in front of your webcam, the application continuously compares your body position against a preprocessed artwork database and displays the closest match.

The interface presents three synchronized views:
    1. Live Video: Webcam feed with real-time MediaPipe skeleton tracking.
    2. Pose Match: The corresponding artwork with its detected skeleton overlay.
    3. Original Artwork: The original artwork without annotations.
    
The system updates its artwork selection every five webcam frames, creating a continuous interaction between physical movement and art discovery.

1. Clone the repository
```bash

git clone https://github.com/YOUR_USERNAME/living-pose-gallery.git
cd living-pose-gallery
```
2. Create a virtual environment
```bash

python -m venv venv
```
Activate it:

macOS / Linux
```bash

source venv/bin/activate
```
Windows
```bash

venv\Scripts\activate
```
3. Install dependencies
```bash

python -m pip install --upgrade pip
python -m pip install flask numpy pillow ultralytics opencv-python
```
Note: MediaPipe does not need to be installed with Python. It runs directly in the browser through the MediaPipe Tasks Vision JavaScript library.

OpenCV is required for the original preprocessing implementation, but not for browser-based live tracking.
### Project Structure

```text
living-pose-gallery/
│
├── pose_gallery.py
├── README.md
│
├── artworks/
│   ├── artwork_01.jpg
│   ├── artwork_02.jpg
│   └── artwork_03.jpg
│
└── cache/
    ├── poses.npz
    │
    ├── images/
    │   ├── 0000_artwork_01.jpg
    │   └── 0001_artwork_02.jpg
    │
    └── overlays/
        ├── 0000_artwork_01__p1.jpg
        └── 0001_artwork_02__p1.jpg
```

### Usage
Run:
```python
pose_gallery.py preprocess ./artworks --cache ./cache --backend yolo
```
This process detects people, extracts their body keypoints and generates the corresponding skeleton overlays.

Preprocessing only needs to be performed once, or whenever the artwork collection changes.

Available preprocessing backends:

    yolo
    openpose

 Launch the application

Run:
bash

```python
pose_gallery.py run --cache ./cache --port 7860
```

Open the application in your browser:

http://127.0.0.1:7860

Click START EXPERIENCE and allow webcam access.
