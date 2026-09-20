"""
transcribe.py — local speech-to-text using faster-whisper.

Loads a Whisper model on CPU and transcribes an audio file to plain text.

Model size tradeoffs (all run on CPU):
  tiny  (~75MB)  — fastest, least accurate, fine for clear audio
  base  (~145MB) — good balance for quick experiments
  small (~244MB) — noticeably better accuracy on accents/noise; chosen default
  medium/large   — much slower on CPU, not practical for local use without GPU
"""

import sys
from faster_whisper import WhisperModel


# Change to "base" or "tiny" if you need faster (but less accurate) transcription.
MODEL_SIZE = "small"


def load_model(model_size: str = MODEL_SIZE) -> WhisperModel:
    """Load and return a faster-whisper model running on CPU.

    Args:
        model_size: One of "tiny", "base", "small", "medium", "large".

    Returns:
        A WhisperModel instance ready for transcription.
    """
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(
    file_path: str,
    model: WhisperModel | None = None,
    initial_prompt: str | None = None,
) -> str:
    """Transcribe an audio file to text.

    Args:
        file_path: Absolute or relative path to the audio file (mp3/m4a/wav).
        model: Optional pre-loaded WhisperModel. If None, one is loaded fresh.
        initial_prompt: Optional hint text to bias Whisper toward domain vocabulary.
            Example: "backend, API, frontend, deployment, pull request"
            Useful when transcribing technical speech that Whisper commonly mishears.

    Returns:
        The full transcript as a single string.
    """
    if model is None:
        model = load_model()

    segments, _info = model.transcribe(file_path, beam_size=5, initial_prompt=initial_prompt)

    # segments is a generator — join all segment texts into one string
    transcript = " ".join(segment.text.strip() for segment in segments)
    return transcript


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path-to-audio-file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Loading model ({MODEL_SIZE})...")
    whisper_model = load_model()

    print(f"Transcribing: {audio_path}")
    result = transcribe_audio(audio_path, model=whisper_model)

    print("\n--- TRANSCRIPT ---")
    print(result)
