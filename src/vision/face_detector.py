import cv2
import mediapipe as mp
from .base_detector import BaseDetector


class FaceDetector(BaseDetector):
    """Detects and tracks faces in video frames.

    Events:
        face_appeared: When a new face is detected
        face_disappeared: When a face is no longer detected
        faces_tracked: Every frame with face position data for all faces
    """

    def __init__(
        self, detection_confidence=0.5, tracking_threshold=0.3, debug=False
    ):
        """Initialize the face detector.

        Args:
            detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection.
            tracking_threshold (float): Minimum confidence to continue tracking a face.
            debug (bool): Whether to enable debug mode
        """

        super().__init__()
        self.min_detection_confidence = detection_confidence
        self.tracking_threshold = tracking_threshold
        self.face_detection = None
        self.faces_present = 0
        self.debug = debug
        self.mp_drawing = mp.solutions.drawing_utils

    async def setup(self):
        """Initialize the face detection model.

        Sets up the MediaPipe face detection model with the specified confidence threshold.
        """

        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=1,
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
            - face_appeared: When a new face is detected
            - face_disappeared: When faces are no longer detected
            - faces_tracked: Every frame with face position data for all faces
        """

        results = self.face_detection.process(frame)

        faces_data = []
        current_faces = 0

        if results.detections:
            current_faces = len(results.detections)

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_data = {
                    "x": bbox.xmin + bbox.width / 2,
                    "y": bbox.ymin + bbox.height / 2,
                    "size": bbox.width * bbox.height,
                    "confidence": detection.score[0],
                }
                faces_data.append(face_data)

            if current_faces > self.faces_present:
                self.emit(
                    "face_appeared",
                    {
                        "total_faces": current_faces,
                        "new_faces": current_faces - self.faces_present,
                    },
                )

            if self.debug:
                debug_frame = frame.copy()
                for detection in results.detections:
                    self.mp_drawing.draw_detection(debug_frame, detection)

                height, width = debug_frame.shape[:2]
                max_height = 600
                max_width = 1024

                if height > max_height or width > max_width:
                    scale = min(max_height / height, max_width / width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    debug_frame = cv2.resize(
                        debug_frame, (new_width, new_height)
                    )

                cv2.imshow(
                    "Face Detection Debug",
                    cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)

            self.emit(
                "faces_tracked",
                {"total_faces": current_faces, "faces": faces_data},
            )

        if current_faces < self.faces_present:
            self.emit(
                "face_disappeared",
                {
                    "previous_faces": self.faces_present,
                    "current_faces": current_faces,
                },
            )

        self.faces_present = current_faces
