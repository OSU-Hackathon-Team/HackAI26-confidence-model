import os
import sys
import subprocess
import json

def get_file_json(abs_path, workspace_root):
    """
    Returns a dictionary following the user-provided format.
    """
    rel_path = os.path.relpath(abs_path, workspace_root)
    return {
        "file": {
            "absoluteUri": f"file://{abs_path}",
            "workspaceUrisToRelativePaths": {
                f"file://{workspace_root}": rel_path
            }
        }
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python import_video.py <input_mp4> [title] [confident/unconfident]")
        print("Example: python import_video.py test.mp4 \"My Clip\" confident")
        sys.exit(1)

    input_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        sys.exit(1)

    # Resolve workspace root relative to this script
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(workspace_root, "HackAI26-Training-Data/training_data")
    os.makedirs(output_dir, exist_ok=True)

    # Get title and label
    if len(sys.argv) >= 4:
        title = sys.argv[2]
        label = sys.argv[3].upper()
    else:
        # Match data_recorder.py's interactive style if args are missing
        title = input("Enter session title (e.g. My Recording): ").strip() or "imported_video"
        label_input = input("Is this session CONFIDENT? (y/n): ").strip().lower()
        label = "CONFIDENT" if label_input == 'y' else "UNCONFIDENT"

    # Sanitize title
    safe_title = "".join([c if c.isalnum() else "_" for c in title])
    # Prefix 'I' for Imported to distinguish from 'M' (Manual?) or 'R' (Recorded?)
    base_name = f"I_{safe_title}-{label}-BOTH"
    base_path = os.path.join(output_dir, base_name)

    video_output = f"{base_path}.mp4"
    audio_output = f"{base_path}.wav"

    print(f"\nProcessing: {input_path}")
    print(f"Target Base: {base_name}")

    try:
        # Check available streams using ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", input_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        streams = probe_result.stdout.strip().split("\n")
        
        has_video = "video" in streams
        has_audio = "audio" in streams

        outputs = []

        # 1. Extract Video track if present
        if has_video:
            print("  > Extracting video track...")
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-an", "-c:v", "copy", video_output
            ], check=True, capture_output=True)
            outputs.append(("Video", video_output))
        else:
            print("  > No video track found.")

        # 2. Extract Audio track if present
        if has_audio:
            print("  > Extracting audio track (mono WAV, 44.1kHz)...")
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", audio_output
            ], check=True, capture_output=True)
            outputs.append(("Audio", audio_output))
        else:
            print("  > No audio track found.")

        if not outputs:
            print("\nError: No valid tracks found to extract.")
            sys.exit(1)

        print("\nSUCCESS: Processing complete.")

        # Print the requested JSON format for each generated file
        for label_type, out_path in outputs:
            info = get_file_json(out_path, workspace_root)
            print(f"\n--- {label_type} Metadata ---")
            print(json.dumps(info, indent=2))

        print(f"\nFiles saved to: {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg Error: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
