"""
STEP 2: Convert call center audio files to text transcripts.
Uses OpenAI Whisper (runs locally, completely free).
"""

import whisper
import os
import json
from datetime import datetime


def transcribe_audio_files(audio_folder, output_folder):
    """
    Takes a folder of audio files and produces text transcripts.
    
    Parameters:
        audio_folder: path to folder containing .wav or .mp3 files
        output_folder: path to save the transcripts
    """
    
    # Load the Whisper model
    # Options: "tiny", "base", "small", "medium", "large"
    # "base" is a good balance of speed and accuracy for beginners
    print("Loading Whisper model... (this may take a minute the first time)")
    model = whisper.load_model("base")
    
    # Get list of audio files
    supported_formats = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    audio_files = [
        f for f in os.listdir(audio_folder)
        if f.lower().endswith(supported_formats)
    ]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        print(f"Supported formats: {supported_formats}")
        return []
    
    print(f"Found {len(audio_files)} audio file(s) to transcribe.\n")
    
    all_transcripts = []
    
    for i, filename in enumerate(audio_files, 1):
        filepath = os.path.join(audio_folder, filename)
        print(f"[{i}/{len(audio_files)}] Transcribing: {filename}")
        
        try:
            # This is where the magic happens — Whisper converts audio to text
            result = model.transcribe(filepath)
            
            transcript = {
                "file": filename,
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"]
                    }
                    for seg in result.get("segments", [])
                ],
                "transcribed_at": datetime.now().isoformat()
            }
            
            all_transcripts.append(transcript)
            
            # Show a preview of the first 200 characters
            preview = result["text"][:200]
            print(f"   Preview: {preview}...")
            print(f"   Language detected: {result.get('language', 'unknown')}\n")
            
        except Exception as e:
            print(f"   ERROR transcribing {filename}: {e}\n")
    
    # Save all transcripts to a JSON file
    output_path = os.path.join(output_folder, "transcripts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! {len(all_transcripts)} transcripts saved to {output_path}")
    return all_transcripts


# ---- RUN THIS SCRIPT ----
if __name__ == "__main__":
    # Points to your raw audio files
    AUDIO_FOLDER = "data/audio/raw"
    
    # Saves processed transcripts here
    OUTPUT_FOLDER = "data/transcripts/processed"
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    transcripts = transcribe_audio_files(AUDIO_FOLDER, OUTPUT_FOLDER)