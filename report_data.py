import os
import glob
import cv2
import wave
import contextlib
import pickle
from collections import defaultdict

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return duration, frames

def get_audio_info(path):
    try:
        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception:
        return 0

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

def main():
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_root, "HackAI26-Training-Data/training_data")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    # Find all unique session bases
    all_files = glob.glob(os.path.join(data_dir, "*"))
    sessions = defaultdict(dict)
    
    for f in all_files:
        base, ext = os.path.splitext(f)
        session_name = os.path.basename(base)
        if ext in ['.mp4', '.wav', '.pkl']:
            sessions[session_name][ext] = f

    # stats[label][category] = {count, duration, frames}
    # category is "VIDEO" or "AUDIO"
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "duration": 0, "frames": 0}))
    
    total_count = 0
    total_duration = 0
    total_frames = 0

    print("Analyzing training data variety (Video vs Audio stats)...\n")

    for name, files in sessions.items():
        try:
            parts = name.split('-')
            if len(parts) < 3:
                continue
            
            label = parts[-2].upper()
            source_type = parts[-1].upper()
        except Exception:
            label = "UNKNOWN"
            source_type = "UNKNOWN"

        video_duration, video_frames = 0, 0
        audio_duration = 0

        if '.mp4' in files:
            video_duration, video_frames = get_video_info(files['.mp4'])
        if '.wav' in files:
            audio_duration = get_audio_info(files['.wav'])

        # Mapping BOTH -> (VIDEO and AUDIO)
        target_categories = []
        if source_type == "BOTH":
            target_categories = ["VIDEO", "AUDIO"]
        elif source_type == "VIDEO":
            target_categories = ["VIDEO"]
        elif source_type == "AUDIO":
            target_categories = ["AUDIO"]

        for cat in target_categories:
            if cat == "VIDEO":
                stats[label][cat]["count"] += 1
                stats[label][cat]["duration"] += video_duration
                stats[label][cat]["frames"] += video_frames
            else: # AUDIO
                stats[label][cat]["count"] += 1
                # If it's BOTH or AUDIO, we take the audio duration
                stats[label][cat]["duration"] += (audio_duration if audio_duration > 0 else video_duration)
                stats[label][cat]["frames"] += video_frames # Audio recordings don't have frames, but we track the context frames if BOTH

        total_count += 1
        total_duration += video_duration if video_duration > 0 else audio_duration
        total_frames += video_frames

    # Print Report
    header = f"{'Variety':<20} | {'Count':<8} | {'Duration':<12} | {'Frames':<10}"
    print(header)
    print("-" * len(header))

    labels = sorted(stats.keys())
    for label in labels:
        # Calculate label totals
        l_count, l_dur, l_frames = 0, 0, 0
        seen_sessions = set()
        
        # We need to recalculate label totals carefully because one session might be in two cats
        # But for 'label' total we just care about unique sessions with that label
        label_sessions = [name for name, files in sessions.items() if name.split('-')[-2].upper() == label]
        l_count = len(label_sessions)
        
        for name in label_sessions:
            files = sessions[name]
            vd, vf = 0, 0
            if '.mp4' in files:
                vd, vf = get_video_info(files['.mp4'])
            ad = get_audio_info(files['.wav']) if '.wav' in files else 0
            
            l_dur += max(vd, ad)
            l_frames += vf

        for cat in ["VIDEO", "AUDIO"]:
            s = stats[label][cat]
            if s['count'] == 0: continue
            
            variety = f"{label}-{cat}"
            p_count = (s['count'] / total_count * 100) if total_count > 0 else 0
            p_dur = (s['duration'] / total_duration * 100) if total_duration > 0 else 0
            p_frames = (s['frames'] / total_frames * 100) if total_frames > 0 else 0

            line = (f"{variety:<20} | "
                    f"{s['count']:>2} ({p_count:2.0f}%) | "
                    f"{format_time(s['duration']):<7} ({p_dur:2.0f}%) | "
                    f"{s['frames']:>5} ({p_frames:2.0f}%)")
            print(line)
        
        # Print Label Summary
        p_l_count = (l_count / total_count * 100) if total_count > 0 else 0
        p_l_dur = (l_dur / total_duration * 100) if total_duration > 0 else 0
        p_l_frames = (l_frames / total_frames * 100) if total_frames > 0 else 0
        
        print(f"{'TOTAL ' + label:<20} | "
              f"{l_count:>2} ({p_l_count:2.0f}%) | "
              f"{format_time(l_dur):<7} ({p_l_dur:2.0f}%) | "
              f"{l_frames:>5} ({p_l_frames:2.0f}%)")
        print("-" * len(header))

    print(f"{'TOTAL SESSIONS':<20} | {total_count:<8} | {format_time(total_duration):<12} | {total_frames:<10}")

if __name__ == "__main__":
    main()
