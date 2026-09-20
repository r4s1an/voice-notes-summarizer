"""
main.py — FastAPI backend for the voice notes summarizer.

Exposes a single endpoint:
  POST /process  — accepts an audio file, returns transcript + summary + action items.
"""

import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.transcribe import load_model, transcribe_audio
from app.summarize import get_groq_client, summarize_transcript

load_dotenv()

app = FastAPI(title="Voice Notes Summarizer")

# Load the Whisper model once at startup so every request reuses it.
whisper_model = load_model()
groq_client = get_groq_client()


@app.get("/health")
def health():
    """Simple health check — confirms the server is running."""
    return {"status": "ok"}


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    initial_prompt: str | None = None,
) -> JSONResponse:
    """Accept an audio file, transcribe it, then summarize the transcript.

    Args:
        file: The uploaded audio file (mp3, m4a, or wav).

    Returns:
        JSON with keys: transcript, summary, action_items.
        On error: JSON with key: error.
    """
    allowed_types = {"audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav", "audio/m4a", "audio/x-m4a"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use mp3, m4a, or wav.",
        )

    # Save the upload to a temp file so faster-whisper can read it from disk.
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path, model=whisper_model, initial_prompt=initial_prompt)
    except Exception as exc:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    os.unlink(tmp_path)  # clean up temp file after transcription

    result = summarize_transcript(transcript, client=groq_client)

    if "error" in result:
        raise HTTPException(status_code=429, detail=result["error"])

    return JSONResponse({
        "transcript": transcript,
        "summary": result["summary"],
        "action_items": result["action_items"],
    })
