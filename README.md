
# Hand Gesture Volume Control using MediaPipe and OpenCV

This project implements a hand gesture-based system volume control using **MediaPipe** for hand tracking and **pycaw** for controlling audio volume on Windows systems. The application uses real-time hand landmark detection and calculates the distance between the thumb and index fingertips to adjust the system's volume.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [How It Works](#how-it-works)
- [Code Breakdown](#code-breakdown)
- [How to Run](#how-to-run)
- [Demo](#demo)
- [References](##References)

## Overview

This project leverages **OpenCV** for real-time video capture, **MediaPipe** for hand detection, and **pycaw** for controlling the system volume. The script detects the distance between the thumb and index finger in each video frame, then maps that distance to the system volume range.

The script consists of the following steps:
1. Capturing video from the webcam.
2. Detecting and drawing hand landmarks.
3. Calculating the distance between the thumb and index finger.
4. Mapping the distance to control the system volume.
5. Providing real-time feedback through a volume bar on the screen.

## Prerequisites

Make sure to have Python 3.x installed on your system. All the required libraries are listed in the `requirements.txt` file of this repository. You can install them with the following command:

```bash
pip install -r requirements.txt
```

### Dependencies

- `opencv-python`: For capturing video feed from the webcam and rendering the interface.
- `mediapipe`: For detecting hand landmarks.
- `numpy`: For numerical calculations.
- `pycaw`: For controlling the system audio.
- `comtypes`: To handle Windows audio interfaces via pycaw.

## How It Works

The system uses **MediaPipe's hand landmark detection** to track the position of the thumb and index finger. Based on the distance between these fingers, the system adjusts the system volume in real time. The script provides visual feedback through a webcam window, where it shows the hand landmarks, a line between the thumb and index finger, and a dynamically updating volume bar.

## Code Breakdown

### 1. **Importing Libraries**

```python
import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm
```

- **cv2 (OpenCV)**: Used for accessing the webcam and displaying video feed.
- **numpy**: Provides helper functions for calculating distances between points.
- **pycaw**: Controls system volume by interfacing with Windows audio utilities.
- **HandTrackingModule**: A custom module for hand landmark detection using **MediaPipe**.

### 2. **Audio Setup**

```python
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
min_vol, max_vol = volume.GetVolumeRange()[:2]
```

- **AudioUtilities.GetSpeakers()**: Retrieves the speaker devices connected to the system.
- **IAudioEndpointVolume**: Controls the system volume by adjusting its levels.
- The volume range is fetched using `GetVolumeRange()` and mapped to the distance between the thumb and index finger.

### 3. **Initializing Webcam and Hand Detector**

```python
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Couldn't open the webcam")

detector = htm.HandDetector(detectionCon=0.7, trackCon=0.7)
vol, vol_bar, vol_per = 0, 400, 0
```

- **cv.VideoCapture(0)**: Captures video frames from the default webcam.
- **HandDetector**: A custom class from `HandTrackingModule` responsible for detecting hands and returning landmarks.
- **detectionCon** and **trackCon**: Confidence levels for hand detection and tracking, respectively.
- **vol**, **vol_bar**, and **vol_per** are initialized for controlling volume, the visual volume bar, and volume percentage.

### 4. **Main Loop for Frame Processing**

```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.findHands(frame, draw=True)
    landmarks = detector.findPositions(frame)
```

- The loop runs continuously, capturing video frames.
- **findHands()** detects hands in the frame, and **findPositions()** returns the hand landmarks (key points of the hand).

### 5. **Gesture-Based Volume Control**

```python
if landmarks:
    x1, y1 = landmarks[4][:2]  # Thumb tip
    x2, y2 = landmarks[8][:2]  # Index finger tip
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Visualizing landmarks
    for (x, y) in [(x1, y1), (x2, y2), (cx, cy)]:
        cv.circle(frame, (x, y), 10, (255, 0, 255) if (x, y) != (cx, cy) else (0, 255, 0), cv.FILLED)
    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    length = np.linalg.norm([x2 - x1, y2 - y1])
    vol = np.interp(length, [20, 200], [min_vol, max_vol])
    vol_bar = np.interp(length, [20, 200], [400, 150])
    vol_per = np.interp(length, [20, 200], [0, 100])

    volume.SetMasterVolumeLevel(vol, None)
```

- **landmarks[4][:2]** and **landmarks[8][:2]**: Get the (x, y) coordinates for the thumb and index finger tips.
- The distance between the thumb and index finger is calculated using `np.linalg.norm()`.
- **np.interp()**: Maps the finger distance to the system's volume range. The shorter the distance, the lower the volume, and vice versa.

### 6. **Displaying Volume Bar and Percentage**

```python
bar_color = (0, 255, 0) if vol_per <= 70 else (0, 0, 255)
cv.rectangle(frame, (50, 150), (85, 400), bar_color, 3)
cv.rectangle(frame, (50, int(vol_bar)), (85, 400), bar_color, cv.FILLED)
cv.putText(frame, f"{int(vol_per)} %", (45, 140), cv.FONT_HERSHEY_COMPLEX, 1.25, (255, 255, 255), 2)
```

- **cv.rectangle()**: Draws the volume bar and fills it based on the current volume percentage.
- **cv.putText()**: Displays the current volume percentage on the screen.

### 7. **Ending the Program**

```python
if cv.waitKey(1) & 0xFF == ord('p'):
    break
```

- The program runs in a loop until the user presses the 'p' key, which breaks the loop and ends the script.

### 8. **Resource Cleanup**

```python
cap.release()
cv.destroyAllWindows()
```

- **cap.release()**: Releases the webcam.
- **cv.destroyAllWindows()**: Closes any OpenCV windows that were opened during the program.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-gesture-volume-control.git
   cd hand-gesture-volume-control
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python hand_volume_control.py
   ```

4. Use your webcam to control the system volume by moving your thumb and index finger. Spread them apart to increase the volume, and pinch them to lower it.

5. Press the `p` key to exit the application.

## Demo

Here’s a demo of the application in action:

## References
1. [Mediapipe's article on Hand Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)


---
