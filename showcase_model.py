import cv2
import time
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.interpolate import interp1d
from models import VisualConfidenceModel

# --- CONFIGURATION ---
MODEL_PATH = "visual_confidence.pth"
FACE_TASK_PATH = "HackAI26-Training-Data/face_landmarker.task"
HAND_TASK_PATH = "HackAI26-Training-Data/hand_landmarker.task"
INPUT_DIM = 178
SEQUENCE_LENGTH = 30
WINDOW_SIZE_MS = 1000
GRAPH_WINDOW = 100 # Number of confidence points to show in graph

class ShowcaseApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Model
        self.model = VisualConfidenceModel(input_dim=INPUT_DIM)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print(f"Loaded model weights from {MODEL_PATH}")
        else:
            print(f"Warning: {MODEL_PATH} not found. Using untrained weights.")
        self.model.to(self.device)
        self.model.eval()

        # Initialize MediaPipe
        self.init_mediapipe()

        # Data Buffers
        self.feature_history = [] # List of (timestamp_ms, feature_vector)
        self.confidence_history = [0.5] * GRAPH_WINDOW
        
        # UI State
        self.last_inference_time = 0
        self.inference_interval = 0.1 # Run inference every 100ms
        self.current_confidence = 0.5

    def init_mediapipe(self):
        print("Initializing MediaPipe...")
        # Face Landmarker
        face_base_options = python.BaseOptions(model_asset_path=FACE_TASK_PATH)
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            output_face_blendshapes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # Hand Landmarker
        hand_base_options = python.BaseOptions(model_asset_path=HAND_TASK_PATH)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)


    def run_inference(self):
        if len(self.feature_history) < 10:
            return 0.5

        # Extract features for the last 1s
        now_ms = self.feature_history[-1][0]
        start_ms = now_ms - WINDOW_SIZE_MS
        
        window_data = [f for t, f in self.feature_history if t >= start_ms]
        window_times = [t for t, f in self.feature_history if t >= start_ms]

        if len(window_data) < 5:
            return 0.5

        # Resample to SEQUENCE_LENGTH steps
        window_data = np.array(window_data)
        window_times = np.array(window_times)
        
        target_ts = np.linspace(window_times[0], window_times[0] + WINDOW_SIZE_MS, SEQUENCE_LENGTH)
        
        try:
            f_interp = interp1d(window_times, window_data, axis=0, kind='linear', fill_value="extrapolate")
            resampled_seq = f_interp(target_ts)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(resampled_seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                # Class 1 is CONFIDENT, Class 0 is UNCONFIDENT
                confidence = probs[0][1].item()
                return confidence
        except Exception as e:
            # print(f"Inference error: {e}")
            return 0.5

    def draw_ui(self, frame, face_result, hand_result):
        h, w, _ = frame.shape
        
        # 1. LANDMARK VISUALIZATION
        # (Optional but adds 'cool' factor to showcase)
        if face_result and face_result.face_landmarks:
            for i, lm in enumerate(face_result.face_landmarks[0]):
                if i % 10 == 0: # Sparse for performance
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 1, (0, 255, 255), -1)

        if hand_result and hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                for lm in hand_lms:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # 2. OVERLAY HEADER
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 3. CONFIDENCE GAUGE (Top Left)
        gauge_w = 200
        gauge_h = 20
        gx, gy = 20, 45
        # Background
        cv2.rectangle(frame, (gx, gy), (gx + gauge_w, gy + gauge_h), (50, 50, 50), -1)
        # Fill
        fill_w = int(self.current_confidence * gauge_w)
        color = (0, 255, 0) if self.current_confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (gx, gy), (gx + fill_w, gy + gauge_h), color, -1)
        
        label = "CONFIDENT" if self.current_confidence > 0.5 else "UNCONFIDENT"
        cv2.putText(frame, f"{label}: {self.current_confidence:.1%}", (gx, gy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. LIVE GRAPH (Bottom)
        graph_h = 80
        graph_w = w - 40
        graph_x = 20
        graph_y = h - 20 - graph_h
        
        # Background for graph
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Center line (0.5)
        cv2.line(frame, (graph_x, graph_y + graph_h//2), (graph_x + graph_w, graph_y + graph_h//2), (100, 100, 100), 1)

        # Graph lines
        if len(self.confidence_history) > 1:
            pts = []
            for i, conf in enumerate(self.confidence_history):
                px = graph_x + int(i * (graph_w / (GRAPH_WINDOW - 1)))
                py = graph_y + graph_h - int(conf * graph_h)
                pts.append([px, py])
            
            pts = np.array(pts, np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        # 5. Pulsating LIVE dot
        dot_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 0, 100)
        cv2.circle(frame, (w - 30, 30), 8, dot_color, -1)
        cv2.putText(frame, "LIVE SHOWCASE", (w - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("\nShowcase running! Press 'q' to quit.")
        
        start_time = time.time()
        last_results = (None, None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for natural feel
            frame = cv2.flip(frame, 1)
            
            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)

            # Extract Features
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                last_results = (face_result, hand_result)
                
                feat = self.process_mediapipe_results(face_result, hand_result)
                if feat is not None:
                    self.feature_history.append((timestamp_ms, feat))
            except Exception as e:
                print(f"MediaPipe Error: {e}")

            # Keep buffer lean
            while self.feature_history and self.feature_history[0][0] < timestamp_ms - 2000:
                self.feature_history.pop(0)

            # Periodic Inference
            if current_time - self.last_inference_time > self.inference_interval:
                self.current_confidence = self.run_inference()
                self.confidence_history.append(self.current_confidence)
                if len(self.confidence_history) > GRAPH_WINDOW:
                    self.confidence_history.pop(0)
                self.last_inference_time = current_time

            # UI
            self.draw_ui(frame, last_results[0], last_results[1])
            
            cv2.imshow("Confidence Model Showcase", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()
        self.hand_landmarker.close()

    def process_mediapipe_results(self, face_result, hand_result):
        # Blendshapes
        bs = [0.0] * 52
        if face_result.face_blendshapes:
            bs = [b.score for b in face_result.face_blendshapes[0]]

        # Hands
        lh = np.zeros(63)
        rh = np.zeros(63)
        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                if i < len(hand_result.handedness):
                    side = hand_result.handedness[i][0].category_name
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]).flatten()
                    if side.lower() == 'left':
                        lh = landmarks
                    else:
                        rh = landmarks

        return np.concatenate([bs, lh, rh])

if __name__ == "__main__":
    app = ShowcaseApp()
    app.run()
