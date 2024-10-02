import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm

# Initializing Audio Interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
min_vol, max_vol = volume.GetVolumeRange()[:2]

# Initializing Webcam and Hand Detector
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Couldn't open the webcam")

detector = htm.HandDetector(detectionCon=0.7, trackCon=0.7)
vol, vol_bar, vol_per = 0, 400, 0

while cap.isOpened():
    # Reading frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting hands and finding positions
    frame = detector.findHands(frame, draw=True)
    landmarks = detector.findPositions(frame)

    if landmarks:
        # Getting coordinates of thumb and index fingertips
        x1, y1 = landmarks[4][:2]
        x2, y2 = landmarks[8][:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Visualizing landmarks and connection
        for (x, y) in [(x1, y1), (x2, y2), (cx, cy)]:
            # Drawing circles for fingertips and center
            cv.circle(frame, (x, y), 10, (255, 0, 255) if (x, y) != (cx, cy) else (0, 255, 0), cv.FILLED)
        # Drawing line between thumb and index finger
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Calculating distance between thumb and index finger and mapping it to volume range
        length = np.linalg.norm([x2 - x1, y2 - y1])
        vol = np.interp(length, [20, 200], [min_vol, max_vol])
        vol_bar = np.interp(length, [20, 200], [400, 150])
        vol_per = np.interp(length, [20, 200], [0, 100])

        # Setting system volume level based on hand distance
        volume.SetMasterVolumeLevel(vol, None)

    # Flipping frame horizontally for mirror effect
    frame = cv.flip(frame, 1)

    # Determining color for volume bar based on percentage
    bar_color = (0, 255, 0) if vol_per <= 70 else (0, 0, 255)

    # Drawing volume bar and displaying percentage
    cv.rectangle(frame, (50, 150), (85, 400), bar_color, 3)
    cv.rectangle(frame, (50, int(vol_bar)), (85, 400), bar_color, cv.FILLED)
    cv.putText(frame, f"{int(vol_per)} %", (45, 140), cv.FONT_HERSHEY_COMPLEX, 1.25, (255, 255, 255), 2)

    # Displaying frame
    cv.imshow("Hand", frame)

    # Breaking loop on pressing 'p'
    if cv.waitKey(1) & 0xFF == ord('p'):
        break

# Releasing webcam and closing windows
cap.release()
cv.destroyAllWindows()
