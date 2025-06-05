import argparse
import os
import whisper
from pydub import AudioSegment

def convert_to_wav(audio_path):
    print(f"[INFO] Converting {audio_path} to WAV format...")
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_path, model_size="base"):
    print(f"[INFO] Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    print(f"[INFO] Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    return result["text"]

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument("file", help="Path to the MP3 or M4A file")
    parser.add_argument("--model", default="base", help="Whisper model size: tiny, base, small, medium, large")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("[ERROR] File not found.")
        return

    if not args.file.lower().endswith((".mp3", ".m4a")):
        print("[ERROR] File must be an MP3 or M4A.")
        return

    wav_file = convert_to_wav(args.file)
    transcript = transcribe_audio(wav_file, model_size=args.model)

    output_file = os.path.splitext(args.file)[0] + "_transcript.txt"
    with open(output_file, "w") as f:
        f.write(transcript)

    print(f"[SUCCESS] Transcription saved to {output_file}")

if __name__ == "__main__":
    main()
