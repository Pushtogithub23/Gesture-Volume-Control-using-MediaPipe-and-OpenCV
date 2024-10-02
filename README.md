
# Hand Gesture Volume Control

This repository contains a Python script that controls the system volume using hand gestures, leveraging **MediaPipe** for hand tracking and **pycaw** for controlling audio. The script detects hand landmarks (thumb and index finger), calculates the distance between them, and maps it to the system's audio volume level.

## Features

- **Hand Detection**: Uses MediaPipe's Hand Landmark Detection to track the hand.
- **Volume Control**: Adjusts the system's volume based on the distance between the thumb and index finger.
- **Real-Time Feedback**: Displays a volume bar that dynamically changes based on hand gestures.
- **Simple Interface**: Uses OpenCV to show hand landmarks and the volume control bar in a video feed.

## Prerequisites

To run this project, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencies

The required dependencies are already defined in the `requirements.txt` file. They include:

- `mediapipe`
- `opencv-python`
- `numpy`
- `pycaw`
- `comtypes`

Ensure that you have **Python 3.x** installed before proceeding.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Pushtogithub23/Gesture-Volume-Control-using-MediaPipe-and-OpenCV.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python hand_volume_control.py
   ```

4. Use your webcam to control the system volume by adjusting the distance between your thumb and index finger. Pinch your fingers to lower the volume or spread them apart to increase it.

5. Press the `p` key to stop the application.

## Code Explanation

This section explains the key parts of the script in detail:

### 1. **Importing Required Libraries**

```python
import mediapipe as mp
import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import Hand_trackingModule as htm
```

- **mediapipe**: Used for detecting hand landmarks.
- **opencv-python (cv2)**: Used to capture video frames from the webcam and display the output.
- **numpy**: Helps in numerical operations like calculating the distance between points.
- **pycaw**: A Python interface to control system audio.
- **Hand_trackingModule**: Your custom module that handles hand detection and landmark retrieval.

### 2. **Setting Up the Audio Interface**

```python
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
min_vol, max_vol = volume.GetVolumeRange()[:2]
```

- **AudioUtilities** and **IAudioEndpointVolume** from `pycaw` are used to get access to the system's audio settings.
- `GetVolumeRange()` returns the minimum and maximum volume levels supported by the system. These values are used later to map hand gesture distances to the volume range.

### 3. **Initializing Webcam and Hand Detector**

```python
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Couldn't open the webcam")

detector = htm.HandDetector()
vol, vol_bar, vol_per = 0, 400, 0
```

- The webcam is accessed using OpenCV's `VideoCapture(0)`, which reads frames from the camera.
- A custom `HandDetector` class from `Hand_trackingModule` is used to detect hand landmarks.
- `vol`, `vol_bar`, and `vol_per` are initialized to store the current volume level, volume bar height, and volume percentage respectively.

### 4. **Main Loop for Video Frame Processing**

```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.findHands(frame, draw=False)
    landmarks = detector.findPositions(frame)
```

- The main loop reads frames from the webcam, and the `findHands` method detects hands in each frame.
- `findPositions` returns a list of hand landmarks (if a hand is detected).

### 5. **Processing Hand Gestures**

```python
if landmarks:
    x1, y1 = landmarks[4][:2]  # Thumb tip
    x2, y2 = landmarks[8][:2]  # Index finger tip
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    length = np.linalg.norm([x2 - x1, y2 - y1])
    vol = np.interp(length, [20, 200], [min_vol, max_vol])
    vol_bar = np.interp(length, [20, 200], [400, 150])
    vol_per = np.interp(length, [20, 200], [0, 100])
```

- **Thumb and Index Finger Coordinates**: The code retrieves the (x, y) coordinates of the thumb tip (landmark 4) and index fingertip (landmark 8).
- **Calculating Distance**: The distance between the thumb and index finger is calculated using `np.linalg.norm()`.
- **Mapping Distance to Volume**: The distance is mapped to a volume range using `np.interp()`. The same is done for the volume bar and percentage values.

### 6. **Setting System Volume**

```python
volume.SetMasterVolumeLevel(vol, None)
```

- The system volume is adjusted based on the calculated distance between the thumb and index finger using `SetMasterVolumeLevel()`.

### 7. **Drawing Visual Feedback**

```python
frame = cv.flip(frame, 1)  # Mirror the frame

bar_color = (0, 255, 0) if vol_per <= 70 else (0, 0, 255)
cv.rectangle(frame, (50, 150), (85, 400), bar_color, 3)
cv.rectangle(frame, (50, int(vol_bar)), (85, 400), bar_color, cv.FILLED)
cv.putText(frame, f"{int(vol_per)} %", (45, 140), cv.FONT_HERSHEY_COMPLEX, 1.25, (255, 255, 255), 2)
```

- The frame is flipped horizontally for a mirror effect so that the video feed acts like a mirror.
- A volume bar is drawn on the screen with a color that changes based on the current volume percentage.
- The volume percentage is displayed as text on the screen.

### 8. **Handling User Input and Exiting**

```python
if cv.waitKey(1) & 0xFF == ord('p'):
    break
```

- The loop runs continuously until the user presses the 'p' key, which stops the program.

### 9. **Releasing Resources**

```python
cap.release()
cv.destroyAllWindows()
```

- The webcam is released, and all OpenCV windows are closed when the program ends.

## Demo

Here’s a quick demo of how the application looks in action:


## Customization

- You can modify the hand-tracking behaviour by editing the `Hand_trackingModule` or tweaking the `findHands()` and `findPositions()` functions in the main script.
- Adjust the mapping of hand distances to volume levels by changing the interpolation parameters in the code.


---
