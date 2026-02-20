import cv2
import time
import os
import sys
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def validate_device(device_idx, samplerate, channels):
    try:
        sd.check_input_settings(device=device_idx, samplerate=samplerate, channels=channels)
        with sd.InputStream(device=device_idx, samplerate=samplerate, channels=channels):
            pass
        return True
    except Exception:
        return False


def select_audio_device(fs=44100, channels=1):
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        return None, fs

    valid_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            if validate_device(i, fs, channels):
                valid_devices.append((i, d, fs))
            else:
                fallback_fs = int(d['default_samplerate'])
                if validate_device(i, fallback_fs, channels):
                    valid_devices.append((i, d, fallback_fs))

    if not valid_devices:
        print("No valid audio input devices found.")
        return None, fs

    print("\nAvailable Valid Audio Input Devices:")
    default_idx = sd.default.device[0]
    valid_indices = [v[0] for v in valid_devices]

    if default_idx not in valid_indices:
        default_idx = valid_devices[0][0]

    for i, d, rate in valid_devices:
        marker = "*" if i == default_idx else " "
        rate_info = f"@{rate}Hz" if rate != fs else ""
        print(f"{marker} {i}: {d['name']} (channels: {d['max_input_channels']}) {rate_info}")

    selection = input(f"\nSelect audio device ID [Enter for {default_idx}]: ").strip()

    if not selection:
        sel_idx = default_idx
    else:
        try:
            sel_idx = int(selection)
            if sel_idx not in valid_indices:
                sel_idx = default_idx
        except ValueError:
            sel_idx = default_idx

    final_fs = next(v[2] for v in valid_devices if v[0] == sel_idx)
    return sel_idx, final_fs



def get_base_path(title, is_confident, source_type):
    base_dir = "HackAI26-Training-Data/training_data"
    safe_title = "".join([c if c.isalnum() else "_" for c in title])
    label = "CONFIDENT" if is_confident else "UNCONFIDENT"
    base_name = f"{safe_title}-{label}-{source_type.upper()}"
    return os.path.join(os.getcwd(), base_dir, base_name)




def record_session(audio_device_id=None, fs=44100):
    print("\n=== New Recording Session ===")
    title = input("Enter session title: ").strip()
    if not title:
        print("Title cannot be empty.")
        return True

    while True:
        label = input("Is this session CONFIDENT? (y/n): ").strip().lower()
        if label in ['y', 'n']:
            is_confident = (label == 'y')
            break

    while True:
        source_type = input("Classify sample source {audio, video, both}: ").strip().lower()
        if source_type in ['audio', 'video', 'both']:
            break

    is_audio = source_type in ['audio', 'both']
    is_video = source_type in ['video', 'both']

    base_output_path = get_base_path(title, is_confident, source_type)
    print(f"\nOutput base path: {base_output_path}")


    if is_audio and not is_video:
        print("\nAudio-only mode")
        print("Press ENTER to start recording.")
        print("Press ENTER again to stop.\n")

        input()
        audio_buffer = []

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            audio_buffer.append(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=fs,
                channels=1,
                callback=audio_callback,
                device=audio_device_id
            )
            stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return True

        print("Recording STARTED...")
        input()
        stream.stop()
        stream.close()
        print("Recording STOPPED.")

        audio_path = f"{base_output_path}.wav"
        wavfile.write(audio_path, fs, np.concatenate(audio_buffer, axis=0))
        print(f"Audio saved: {audio_path}")
        print("Session complete.")
        return True


    print("\nCamera preview opening...")
    print("Press SPACE to start/stop recording.")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return True

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, 30)

    frames_buffer = []
    audio_buffer = []
    recording = False
    recording_start = None
    audio_stream = None

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_buffer.append(indata.copy())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if recording:
            frames_buffer.append(frame)
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            elapsed = time.time() - recording_start
            cv2.putText(display, f"REC {elapsed:.1f}s", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Recorder", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                recording = True
                recording_start = time.time()
                frames_buffer = []
                audio_buffer = []

                if is_audio:
                    audio_stream = sd.InputStream(
                        samplerate=fs,
                        channels=1,
                        callback=audio_callback,
                        device=audio_device_id
                    )
                    audio_stream.start()

                print("Recording STARTED...")
            else:
                break

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    if audio_stream:
        audio_stream.stop()
        audio_stream.close()

    duration = time.time() - recording_start
    fps = len(frames_buffer) / duration if duration > 0 else 30

    if is_video:
        video_path = f"{base_output_path}.mp4"
        out = cv2.VideoWriter(video_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (width, height))
        for f in frames_buffer:
            out.write(f)
        out.release()
        print(f"Video saved: {video_path}")

    if is_audio:
        audio_path = f"{base_output_path}.wav"
        wavfile.write(audio_path, fs, np.concatenate(audio_buffer, axis=0))
        print(f"Audio saved: {audio_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Session complete.")
    return True




if __name__ == "__main__":
    os.makedirs("HackAI26-Training-Data/training_data", exist_ok=True)

    audio_device, fs = select_audio_device()

    while True:
        if not record_session(audio_device, fs):
            break

        if input("\nRecord another session? (y/n): ").strip().lower() != 'y':
            break