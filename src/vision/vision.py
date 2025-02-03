import cv2
import asyncio
import mediapipe as mp
from pyee.asyncio import AsyncIOEventEmitter


class Vision(AsyncIOEventEmitter):
    """Computer vision system that detects and tracks faces using MediaPipe.

    Emits events when faces appear, disappear, and for continuous face tracking.
    Inherits from AsyncIOEventEmitter to provide event handling capabilities.

    Events:
        face_appeared: Emitted when a face is first detected
        face_disappeared: Emitted when a face is no longer detected
        face_tracked: Emitted every frame with face position data
    """

    def __init__(self):
        """Initialize the vision system.

        Sets up the video capture device and face detection model.
        Initializes face tracking state.
        """
        super().__init__()
        self.cap = cv2.VideoCapture(2)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        self.face_present = False  # Track face state

    async def run(self):
        """Main vision processing loop.

        Continuously captures frames from camera and processes them for face detection.
        Emits events based on face detection results.

        Events:
            ready: When vision system starts
            face_appeared: When a face first appears
            face_tracked: Every frame with face data dictionary containing:
                - x: normalized x coordinate of face center
                - y: normalized y coordinate of face center
                - size: normalized area of face bounding box
        """

        while True:

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = self.face_detection.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            if results.detections:
                if not self.face_present:
                    self.emit("face_appeared")
                    self.face_present = True

                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                face_data = {
                    "x": bbox.xmin + bbox.width / 2,
                    "y": bbox.ymin + bbox.height / 2,
                    "size": bbox.width * bbox.height,
                }
                self.emit("face_tracked", face_data)
            else:
                if self.face_present:
                    self.emit("face_disappeared")
                    self.face_present = False

            await asyncio.sleep(0.1)
