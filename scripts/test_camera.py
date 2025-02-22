import cv2
import time
import numpy as np
import mediapipe as mp

class DualCameraView:
    def __init__(self, width=1024, height=600):
        self.width = width
        self.height = height
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.8,
            model_selection=0
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=4
        )
        self.faces_present = 0
        
        self.left_cap = cv2.VideoCapture('/dev/video45')
        self.right_cap = cv2.VideoCapture('/dev/video46')
        
        self.window_created = False
        self.frame_interval = 1/30
        self.last_frame_time = 0
        
        # Track the last volume to avoid too frequent updates
        self.last_volume = 0
        self.volume_update_threshold = 0.05  # 5% change threshold
        
        for cap in [self.left_cap, self.right_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width // 2)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def create_window(self):
        """Create the OpenCV window for display."""

        if not self.window_created:
            cv2.namedWindow('Dual Camera View', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Dual Camera View', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow('Dual Camera View', self.width, self.height)
            cv2.setWindowProperty('Dual Camera View', cv2.WND_PROP_TOPMOST, 1)
            cv2.displayOverlay('Dual Camera View', '', 1)
            self.window_created = True

    def draw_detection_with_label(self, frame, detection, index):
        """Draw face detection box and label."""

        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        confidence = round(detection.score[0] * 100)
        label = f"Face {index + 1} ({confidence}%)"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        cv2.rectangle(frame, 
                     (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), 
                     (0, 255, 0), 
                     cv2.FILLED)
        
        cv2.putText(frame, 
                    label, 
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 0), 
                    2)
            
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks with connections."""
        
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
            
    def calculate_volume_from_hand(self, hand_landmarks, frame_height):
        """Calculate volume (0-1) based on hand position"""
        # Use the position of the middle finger tip (landmark 12)
        middle_finger_y = hand_landmarks.landmark[12].y
        
        # Convert to relative position (0-1), flip so raising hand increases volume
        volume = 1 - middle_finger_y
        
        # Clamp between 0 and 1
        return max(0, min(1, volume))

    def process_frame(self, frame, detect_features=False):
        """Process a single frame with optional face and hand detection"""
        if frame is None:
            return np.zeros((self.height, self.width // 2, 3), dtype=np.uint8)
            
        # Flip frame for correct orientation
        frame = cv2.flip(frame, 0)
        
        if detect_features:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face detection
            face_results = self.face_detection.process(rgb_frame)
            current_faces = len(face_results.detections) if face_results.detections else 0
            
            if face_results.detections:
                for idx, detection in enumerate(face_results.detections):
                    self.draw_detection_with_label(frame, detection, idx)
                    
            self.faces_present = current_faces
            
            # Process hand detection
            hand_results = self.hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.draw_hand_landmarks(frame, hand_landmarks)
                    
                    # Calculate volume from first detected hand
                    new_volume = self.calculate_volume_from_hand(hand_results.multi_hand_landmarks[0], frame.shape[0])
                    
                    # Only update if change is significant
                    if abs(new_volume - self.last_volume) > self.volume_update_threshold:
                        self.last_volume = new_volume
                        
                        # Make tool call to set volume
                        yield {
                            "type": "function",
                            "function": {
                                "name": "set_volume",
                                "arguments": {
                                    "volume": round(new_volume, 2)
                                }
                            }
                        }
        
        frame = cv2.resize(frame, (self.width // 2, self.height))
        return frame
        
    def run(self):
        """Main loop to display both camera feeds"""
        
        self.create_window()
        
        while True:
            current_time = time.time()
            
            if current_time - self.last_frame_time < self.frame_interval:
                continue
                
            self.last_frame_time = current_time
            
            ret1, left_frame = self.left_cap.read()
            ret2, right_frame = self.right_cap.read()
            
            if not ret1 or not ret2:
                print("Failed to grab frame from one or both cameras")
                break
                
            # Handle tool calls from process_frame
            for tool_call in self.process_frame(left_frame, detect_features=True):
                yield tool_call
                
            right_frame = self.process_frame(right_frame)
            combined_frame = np.hstack((left_frame, right_frame))
            
            cv2.imshow('Dual Camera View', combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.left_cap.release()
        self.right_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        viewer = DualCameraView()
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")