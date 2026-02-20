
import cv2
import sys
import os
import glob
import numpy as np
import mediapipe as mp
import pickle
import whisper
import torch

# --- SETUP MEDIAPIPE ---
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks import python
except ImportError:
    print("Error: MediaPipe Tasks API not found. Please follow installation guide.")
    sys.exit(1)

class DataProcessor:
    """
    Reads .mp4 and .wav pairs in `training_data/`,
    extracts Face/Hand landmarks and Whisper audio embeddings,
    and saves the result to `[filename].pkl`.
    """
    def __init__(self, data_root="HackAI26-Training-Data/training_data"):
        self.data_root = data_root
        
        # Check MediaPipe model files
        face_model = "HackAI26-Training-Data/face_landmarker.task"
        hand_model = "HackAI26-Training-Data/hand_landmarker.task"
        
        if not os.path.exists(face_model) or not os.path.exists(hand_model):
            print("Error: Task files (face_landmarker.task, hand_landmarker.task) not found.")
            sys.exit(1)
            
        print("Initializing MediaPipe models...")
        self.init_mediapipe()
        
        print("Initializing Whisper model ('tiny')...")
        # 'tiny' is fast and good for embedding extraction for confidence
        self.audio_model = whisper.load_model("tiny")
        self.device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
        print("Device: ", self.device)
        self.audio_model.to(self.device)
        print("Models loaded.")

    def init_mediapipe(self):
        # Face Landmarker
        base_options = python.BaseOptions(model_asset_path='HackAI26-Training-Data/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Hand Landmarker
        hand_base_options = python.BaseOptions(model_asset_path='HackAI26-Training-Data/hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def extract_audio_features(self, audio_path):
        """
        Uses Whisper encoder to get embeddings across the audio.
        Returns a dictionary or array mapped to time.
        """
        print(f"  > Processing audio: {os.path.basename(audio_path)}")
        # Whisper expects 16kHz mono audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio) # Whisper works in 30s chunks
        
        # Get Mel Spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # Get Encoder Features
        with torch.no_grad():
            # encoder(mel) -> [B, Seq_Len, Embedding_Dim]
            # Seq_Len is usually 1500 for 30s (representing 20ms steps)
            embeddings = self.audio_model.encoder(mel.unsqueeze(0))
            embeddings = embeddings.squeeze(0).cpu().numpy()
            print(embeddings)
            
        return embeddings

    def process_session(self, video_path, audio_path):
        frames_list = []
        audio_features = None
        
        # 1. Process Audio if available
        if audio_path and os.path.exists(audio_path):
            audio_features = self.extract_audio_features(audio_path)

        # 2. Process Video if available
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening {video_path}")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"  > Processing video: {os.path.basename(video_path)}")
                
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    timestamp_ms = int(frame_idx * 1000 / fps)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    try:
                        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
                        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                    except Exception as e:
                        face_result = type('obj', (object,), {'face_landmarks': [], 'face_blendshapes': []})
                        hand_result = type('obj', (object,), {'hand_landmarks': [], 'handedness': []})
                    
                    frame_features = {
                        "timestamp_ms": timestamp_ms,
                        "face_landmarks": [],
                        "face_blendshapes": [],
                        "left_hand": [],
                        "right_hand": [],
                        "audio_embedding": None
                    }
                    
                    # Store landmarks
                    if face_result.face_landmarks:
                        frame_features["face_landmarks"] = [[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]]
                    if face_result.face_blendshapes:
                        frame_features["face_blendshapes"] = [b.score for b in face_result.face_blendshapes[0]]
                        
                    if hand_result.hand_landmarks:
                        for i, hand_lms in enumerate(hand_result.hand_landmarks):
                            if i < len(hand_result.handedness):
                                side = hand_result.handedness[i][0].category_name
                                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                                frame_features[f"{side.lower()}_hand"] = landmarks

                    # Align Audio Embedding
                    # Whisper features are at 50Hz (20ms per step)
                    if audio_features is not None:
                        # index = timestamp_ms / 20ms
                        feat_idx = int(timestamp_ms / 20)
                        if feat_idx < len(audio_features):
                            frame_features["audio_embedding"] = audio_features[feat_idx]

                    frames_list.append(frame_features)
                    frame_idx += 1
                    if frame_idx % 30 == 0:
                        print(f"    Processed {frame_idx}/{total_frames} frames...", end='\r')
                
                cap.release()
                print(f"\n    Done video. {len(frames_list)} frames.")

        # 3. Handle Audio-Only sessions
        elif audio_features is not None:
            print("  > Processing as audio-only session.")
            # Map audio embeddings to timestamped entries (50Hz)
            for i, emb in enumerate(audio_features):
                frames_list.append({
                    "timestamp_ms": i * 20,
                    "audio_embedding": emb
                })

        return frames_list

    def reset_mediapipe(self):
        self.face_landmarker.close()
        self.hand_landmarker.close()
        self.init_mediapipe()

    def process_all_data(self):
        """
        Finds all unique base names in training_data/ and processes the pairs.
        """
        # Collect all files
        all_files = glob.glob(os.path.join(self.data_root, "*"))
        base_names = set()
        for f in all_files:
            name, ext = os.path.splitext(f)
            if ext in ['.mp4', '.wav']:
                base_names.add(name)
        
        if not base_names:
            print(f"No .mp4 or .wav files found in {self.data_root}/")
            return

        print(f"Found {len(base_names)} sessions to process.")
        
        for base_path in sorted(list(base_names)):
            cache_path = f"{base_path}.pkl"
            
            if os.path.exists(cache_path):
                print(f"Skipping {os.path.basename(base_path)} (already processed)")
                continue

            video_file = f"{base_path}.mp4"
            audio_file = f"{base_path}.wav"
            
            # Reset MediaPipe for new video tracking
            if os.path.exists(video_file):
                self.reset_mediapipe()
            
            extracted_data = self.process_session(
                video_file if os.path.exists(video_file) else None,
                audio_file if os.path.exists(audio_file) else None
            )
            
            if extracted_data:
                with open(cache_path, "wb") as f:
                    pickle.dump(extracted_data, f)
                print(f"Successfully saved to {os.path.basename(cache_path)}")
                
        print("\nAll processing complete.")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all_data()
