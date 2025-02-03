import mediapipe as mp
from .base_detector import BaseDetector


class FaceDetector(BaseDetector):
    """Detects and tracks faces in video frames.

    Events:
        face_appeared: When a face is first detected
        face_disappeared: When a face is no longer detected
        face_tracked: Every frame with face position data
    """

    def __init__(self, min_detection_confidence=0.5):
        """Initialize the face detector.

        Args:
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection.
        """

        super().__init__()
        self.min_detection_confidence = min_detection_confidence
        self.face_detection = None
        self.face_present = False

    async def setup(self):
        """Initialize the face detection model.

        Sets up the MediaPipe face detection model with the specified confidence threshold.
        """

        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence
        )

    async def cleanup(self):
        """Clean up the face detection model.

        Closes the MediaPipe face detection model and releases any resources.
        """

        if self.face_detection:
            self.face_detection.close()

    async def process_frame(self, frame):
        """Process a single frame for face detection.

        Args:
            frame: CV2 image frame to process.

        Detects faces in the frame and emits relevant events:
            - face_appeared: When a face is first detected.
            - face_disappeared: When a face is no longer detected.
            - face_tracked: Every frame with face position data.
        """

        results = self.face_detection.process(frame)

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
