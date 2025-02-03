import cv2
import asyncio
from pyee.asyncio import AsyncIOEventEmitter
from .face_detector import FaceDetector
from .gesture_detector import GestureDetector


class Vision(AsyncIOEventEmitter):
    """Main vision system that manages multiple detection modules.

    Coordinates frame capture and processing across all detectors.
    Aggregates and relays events from individual detectors.
    """

    def __init__(self, camera_id=2):
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        self.detectors = []
        self.running = False

        self.face_detector = FaceDetector()
        self.gesture_detector = GestureDetector()
        self.detectors.extend([self.face_detector, self.gesture_detector])

        def forward_event(event_name, data=None):
            if data is None:
                self.emit(event_name)
            else:
                self.emit(event_name, data)

        self.face_detector.on(
            "face_appeared", lambda: forward_event("face_appeared")
        )
        self.face_detector.on(
            "face_disappeared", lambda: forward_event("face_disappeared")
        )
        self.face_detector.on(
            "face_tracked", lambda data: forward_event("face_tracked", data)
        )

        self.gesture_detector.on(
            "gesture_detected",
            lambda data: forward_event("gesture_detected", data),
        )

    async def setup(self):
        """Initialize camera and detector resources.

        Opens the camera device and initializes all detection modules.
        Must be called before running the vision system.

        Raises:
            RuntimeError: If camera cannot be opened
            Exception: If detector initialization fails
        """

        self.cap = cv2.VideoCapture(self.camera_id)
        for detector in self.detectors:
            await detector.setup()

    async def cleanup(self):
        """Release all resources.

        Closes camera device and cleans up detector resources.
        Should be called when vision system is no longer needed.
        """

        if self.cap:
            self.cap.release()
        for detector in self.detectors:
            await detector.cleanup()

    async def run(self):
        """Run the main vision processing loop.

        Continuously captures frames from camera and processes them through
        all detection modules. Emits events based on detector results.

        The loop continues until self.running is set to False.

        Events are emitted for:
            - Face detection/tracking
            - Gesture recognition
            - Any other enabled detectors

        Note:
            Calls setup() on start and cleanup() when done
            Processes frames asynchronously to avoid blocking
        """

        await self.setup()
        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame through all detectors
            for detector in self.detectors:
                await detector.process_frame(rgb_frame)

            await asyncio.sleep(0.01)

        await self.cleanup()
