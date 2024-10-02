import mediapipe as mp
import cv2 as cv
import time


class HandDetector:
    """
    HandDetector class uses MediaPipe to detect and track hands in video frames.

    Attributes:
    - mode (bool): Static mode or dynamic mode for the hand detection.
    - maxHands (int): Maximum number of hands to detect.
    - complexity (int): Complexity level of the hand landmarks model.
    - detectionCon (float): Minimum detection confidence threshold.
    - trackCon (float): Minimum tracking confidence threshold.
    """

    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe hands and drawing utilities
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_IDS = [4, 8, 12, 16, 20]  # Indices of fingertips

    def findHands(self, frame, draw=True):
        """
        Detects hands in the provided frame and draws landmarks if specified.

        Args:
        - frame (ndarray): The input image in BGR format.
        - draw (bool): Whether to draw the landmarks on the frame.

        Returns:
        - frame (ndarray): The processed frame with landmarks drawn if specified.
        """
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
        self.results = self.hands.process(frame_rgb)  # Process the RGB frame to detect hands

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def findPositions(self, frame, handNo=0, draw=False):
        """
        Finds and returns the dictionary of landmark positions for the specified hand.

        Args:
        - frame (ndarray): The input image.
        - handNo (int): The index of the hand (default is 0 for the first detected hand).
        - draw (bool): Whether to draw circles on landmarks.

        Returns:
        - landmarks (dict): Dictionary of landmark positions with landmark ID as keys and coordinates as values.
        """
        self.landmarks = {}
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks[id] = (cx, cy)

                if draw:
                    cv.circle(frame, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        return self.landmarks

    def fingersUp(self):
        """
        Determines which fingers are up and returns their status.

        Returns:
        - fingers (list): List of integers (0 or 1) representing if the fingers are up (1) or down (0).
        """
        if len(self.landmarks) == 0:
            return []

        fingers = []

        # Determine if it is a left or right hand
        wrist_x = self.landmarks[0][0]  # Wrist x-coordinate
        thumb_x = self.landmarks[self.tip_IDS[0]][0]  # Thumb tip x-coordinate

        # Check thumb status
        if thumb_x > wrist_x:
            # Right hand (Thumb is to the right of the wrist)
            fingers.append(1 if self.landmarks[self.tip_IDS[0]][0] > self.landmarks[self.tip_IDS[0] - 2][0] else 0)
        else:
            # Left hand (Thumb is to the left of the wrist)
            fingers.append(1 if self.landmarks[self.tip_IDS[0]][0] < self.landmarks[self.tip_IDS[0] - 2][0] else 0)

        # Check the status of the other four fingers
        for ID in range(1, 5):
            fingers.append(1 if self.landmarks[self.tip_IDS[ID]][1] < self.landmarks[self.tip_IDS[ID] - 2][1] else 0)

        return fingers

    def release(self):
        """
        Releases the MediaPipe hand detection resources.
        """
        self.hands.close()


def main():
    pTime = 0
    cap = cv.VideoCapture(0)  # Capture video from the default camera
    detector = HandDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findHands(frame)
        landmarks = detector.findPositions(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        frame = cv.flip(frame, 1)  # Flip frame horizontally
        cv.putText(frame, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Hands', frame)

        # Exit when 'p' key is pressed
        if cv.waitKey(1) & 0xFF == ord('p'):
            break

    # Release resources
    cap.release()
    detector.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
