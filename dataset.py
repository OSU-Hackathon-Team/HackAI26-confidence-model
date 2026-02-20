import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

class ConfidenceDataset(Dataset):
    """
    Loads .pkl files from training_data/ and slices them into 1-second windows.
    Handles variable framerates by resampling to a fixed number of steps per window.
    """
    def __init__(self, data_root="HackAI26-Training-Data/training_data", window_size_ms=1000, steps_per_window=30, mode='visual'):
        self.data_root = data_root
        self.window_size_ms = window_size_ms
        self.steps_per_window = steps_per_window
        self.mode = mode # 'visual' or 'audio'
        
        self.samples = []
        self._load_data()

    def _load_data(self):
        pkl_files = [f for f in os.listdir(self.data_root) if f.endswith('.pkl')]
        
        for file in pkl_files:
            # Parse label from filename: [title]-CONFIDENT-[type].pkl
            is_confident = "CONFIDENT" in file and "UNCONFIDENT" not in file
            label = 1 if is_confident else 0
            
            with open(os.path.join(self.data_root, file), 'rb') as f:
                session_data = pickle.load(f)
            
            if not session_data:
                continue
                
            # Slice into 1-second windows
            # We slide the window (e.g., every 500ms for overlap, or 1000ms for no overlap)
            start_ms = session_data[0].get('timestamp_ms', 0)
            end_ms = session_data[-1].get('timestamp_ms', 0)
            
            # Step through the session in increments
            # For training, overlap helps. Let's use 200ms increments.
            stride_ms = 500 
            
            for current_start in range(start_ms, end_ms - self.window_size_ms, stride_ms):
                window_end = current_start + self.window_size_ms
                
                # Extract frames in this time window
                window_frames = [f for f in session_data if current_start <= f.get('timestamp_ms', 0) < window_end]
                
                if len(window_frames) < 5: # Skip too short windows
                    continue
                
                feature_sequence = self._extract_features(window_frames)
                if feature_sequence is not None:
                    self.samples.append((feature_sequence, label))

        print(f"Loaded {len(self.samples)} {self.mode} samples from {len(pkl_files)} files.")

    def _extract_features(self, frames):
        if self.mode == 'visual':
            return self._extract_visual_features(frames)
        else:
            return self._extract_audio_features(frames)

    def _extract_visual_features(self, frames):
        # Extract blendshapes + hand landmarks
        # Blendshapes: 52, Hand: 21*3*2 = 126. Total = 178
        sequence = []
        timestamps = []
        
        for f in frames:
            # Blendshapes
            bs = f.get('face_blendshapes', [])
            if not bs: bs = [0.0] * 52
            
            # Hands
            lh = np.array(f.get('left_hand', [])).flatten()
            if len(lh) == 0: lh = np.zeros(63)
            rh = np.array(f.get('right_hand', [])).flatten()
            if len(rh) == 0: rh = np.zeros(63)
            
            feat = np.concatenate([bs, lh, rh])
            sequence.append(feat)
            timestamps.append(f['timestamp_ms'])
            
        sequence = np.array(sequence)
        timestamps = np.array(timestamps)
        
        # Resample to fixed steps (e.g. 30 fps -> 30 steps)
        if len(sequence) < 2: return None
        
        target_ts = np.linspace(timestamps[0], timestamps[0] + self.window_size_ms, self.steps_per_window)
        
        # Interpolate each feature dimension
        try:
            f_interp = interp1d(timestamps, sequence, axis=0, kind='linear', fill_value="extrapolate")
            resampled_seq = f_interp(target_ts)
            return torch.FloatTensor(resampled_seq)
        except Exception:
            return None

    def _extract_audio_features(self, frames):
        # Whisper embeddings (already at 50Hz ideally)
        sequence = []
        timestamps = []
        
        for f in frames:
            emb = f.get('audio_embedding')
            if emb is not None:
                sequence.append(emb)
                timestamps.append(f['timestamp_ms'])
        
        if len(sequence) < 5: return None
        
        sequence = np.array(sequence)
        timestamps = np.array(timestamps)
        
        # Audio steps are usually 50 per second (20ms)
        target_steps = 50 
        target_ts = np.linspace(timestamps[0], timestamps[0] + self.window_size_ms, target_steps)
        
        try:
            f_interp = interp1d(timestamps, sequence, axis=0, kind='linear', fill_value="extrapolate")
            resampled_seq = f_interp(target_ts)
            return torch.FloatTensor(resampled_seq)
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Test loading
    print("Testing Visual Dataset...")
    try:
        vis_ds = ConfidenceDataset(mode='visual')
        if len(vis_ds) > 0:
            feat, label = vis_ds[0]
            print(f"Visual sample shape: {feat.shape}, Label: {label}")
    except Exception as e:
        print(f"Visual loading failed (maybe no data yet?): {e}")

    print("\nTesting Audio Dataset...")
    try:
        aud_ds = ConfidenceDataset(mode='audio')
        if len(aud_ds) > 0:
            feat, label = aud_ds[0]
            print(f"Audio sample shape: {feat.shape}, Label: {label}")
    except Exception as e:
        print(f"Audio loading failed: {e}")
