import cv2
import asyncio
import mediapipe as mp
from expression.engine import Engine
from expression.expressions import Neutral, Happy, Sad

# Constants
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = 1024, 600

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def get_face_position(frame, face_detection):
    """Detects face in the frame and returns its normalized position and size."""
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        center_x = bbox.xmin + bbox.width / 2
        center_y = bbox.ymin + bbox.height / 2
        size = (
            bbox.width * bbox.height
        )  # Use the area of the bounding box as a proxy for distance
        # Draw bounding box around the face
        mp_drawing.draw_detection(frame, detection)
        return center_x, center_y, size
    return None, None, None


async def main():
    engine = Engine(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)

    # Start the engine
    asyncio.create_task(engine.run())

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5
    ) as face_detection:
        while engine.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            center_x, center_y, face_size = get_face_position(
                frame, face_detection
            )
            if center_x is not None and center_y is not None:
                x_offset, y_offset = center_x, center_y
                # Normalize and center the offset values around 0
                x_offset = (x_offset - 0.5) * 2
                y_offset = (y_offset - 0.5) * 2
                # Clamp the offset values between -1 and 1
                x_offset = max(-1, min(1, x_offset)) / 4
                y_offset = max(-1, min(1, y_offset)) / 4

                # Calculate scale based on face size
                scale = 1.0 + (
                    0.5 - face_size
                )  # Adjust the scaling factor as needed
                # Clamp the scale between 0.7 and 1.3
                scale = max(0.7, min(1.1, scale))

                current_expression = Neutral(
                    position=(x_offset, y_offset),
                    scale=scale,
                    duration=0.1,
                    transition_duration=0.1,
                )
                await engine.queue_animation(current_expression)

            # Display the frame with the bounding box
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.05)  # Adjust the sleep time as needed

        cap.release()
        cv2.destroyAllWindows()


asyncio.run(main())
