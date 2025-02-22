import cv2
import time
import numpy as np
import mediapipe as mp

class CameraView:
    """Handles displaying raw camera feeds with face detection overlay."""

    def __init__(self, vision_instance, width=1024, height=600):
        self.width = width
        self.height = height
        self.vision = vision_instance
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=0
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=2
        )
        
        self.faces_present = 0 
        
        self.left_display_cap = cv2.VideoCapture('/dev/video46')
        self.window_created = False
        self.frame_interval = 1/30
        self.last_frame_time = 0
        
    def create_window(self):
        """Create the OpenCV window for display."""
        
        if not self.window_created:
            cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Camera View', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow('Camera View', self.width, self.height)
            cv2.setWindowProperty('Camera View', cv2.WND_PROP_TOPMOST, 1)
            cv2.displayOverlay('Camera View', '', 1)
            self.window_created = True
        
    def destroy_window(self):
        """Destroy the OpenCV window."""

        if self.window_created:
            cv2.destroyWindow('Camera View')
            self.window_created = False
        
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
        """Draw hand landmarks and connections."""

        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    def get_frame(self, cap, use_vision_camera=False):
        """Get a frame from camera with face and hand detection overlay."""

        if use_vision_camera:
            ret, frame = self.vision.cap.read()
        else:
            ret, frame = cap.read()

        if self.vision.environment == "pi":
                frame = cv2.flip(frame, 0) 
        
        if not ret:
            return np.zeros((self.height, self.width // 2, 3), dtype=np.uint8)
        
        if not use_vision_camera:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_results = self.face_detection.process(rgb_frame)
            
            current_faces = len(face_results.detections) if face_results.detections else 0
            
            if current_faces > self.faces_present:
                self.vision.emit("face_appeared", {
                    "total_faces": current_faces,
                    "new_faces": current_faces - self.faces_present
                })
            elif current_faces < self.faces_present:
                self.vision.emit("face_disappeared", {
                    "previous_faces": self.faces_present,
                    "current_faces": current_faces
                })
            
            if face_results.detections:
                faces_data = []
                for idx, detection in enumerate(face_results.detections):
                    self.draw_detection_with_label(frame, detection, idx)
                    bbox = detection.location_data.relative_bounding_box
                    faces_data.append({
                        "x": bbox.xmin + bbox.width / 2,
                        "y": bbox.ymin + bbox.height / 2,
                        "size": bbox.width * bbox.height,
                        "confidence": detection.score[0]
                    })
                
                self.vision.emit("faces_tracked", {
                    "total_faces": current_faces,
                    "faces": faces_data
                })
            
            hand_results = self.hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                hands_data = []
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.draw_hand_landmarks(frame, hand_landmarks)
                    
                    h, w, _ = frame.shape
                    cx = sum(lm.x for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                    cy = sum(lm.y for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                    
                    hands_data.append({
                        "x": cx,
                        "y": cy
                    })
                
                self.vision.emit("hands_tracked", {
                    "total_hands": len(hand_results.multi_hand_landmarks),
                    "hands": hands_data
                })
            
            self.faces_present = current_faces
        
        frame = cv2.resize(frame, (self.width // 2, self.height))
        return frame
        
    def display_frames(self):
        """Display both camera feeds and return the combined frame."""
        current_time = time.time()
        
        if current_time - self.last_frame_time < self.frame_interval:
            return None
        
        self.last_frame_time = current_time
        self.create_window()
        
        left_frame = self.get_frame(self.left_display_cap)
        right_frame = self.get_frame(None, use_vision_camera=True)
        
        combined = np.hstack((left_frame, right_frame))
        
        cv2.imshow('Camera View', combined)
        cv2.waitKey(1)
        
        return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
    def cleanup(self):
        """Release camera resources and close windows."""
        
        self.left_display_cap.release()
        self.destroy_window() 
        self.destroy_window() 