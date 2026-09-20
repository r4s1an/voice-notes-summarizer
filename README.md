# Voice Notes Summarizer

![Summarization Eval](https://github.com/YOUR_USERNAME/voice-notes-summarizer/actions/workflows/eval.yml/badge.svg)

A local speech-to-text + AI summarization tool. Upload an audio file, get a transcript and structured summary with action items — all from your own machine.

---

## How it works

1. **faster-whisper** transcribes your audio locally (no data leaves your machine)
2. **Groq API** summarizes the transcript and extracts action items
3. **FastAPI** wires them together behind a single endpoint
4. **Streamlit** gives you a simple UI to upload and view results

---

## Requirements

- Python 3.11+
- ffmpeg installed and on your PATH (required by pydub for audio format conversion)
- A [Groq API key](https://console.groq.com)

### Install ffmpeg on Windows

Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your system PATH, or install via winget:

```
winget install ffmpeg
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the example env file and add your Groq key
copy .env.example .env
# Then open .env and set: GROQ_API_KEY=your_key_here
```

---

## Running the app

You need two terminals, both with the virtualenv active.

**Terminal 1 — FastAPI backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run ui/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running without the UI (curl)

```bash
curl -X POST http://localhost:8000/process -F "file=@your_recording.m4a"
```

Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Running the eval harness

```bash
# Both evals (summarization + WER)
python eval/run_eval.py

# Summarization only (no audio files needed)
python eval/run_eval.py --summ-only

# Transcription WER only (requires audio fixtures — see eval/wer_eval.py)
python eval/run_eval.py --wer-only
```

---

## Whisper model size tradeoffs

The model is set in `app/transcribe.py` via the `MODEL_SIZE` constant.

| Model  | Size   | Speed (CPU) | Accuracy | When to use |
|--------|--------|-------------|----------|-------------|
| tiny   | ~75MB  | Very fast   | Low      | Prototyping, clear studio audio |
| base   | ~145MB | Fast        | Moderate | Quick experiments, good mic |
| small  | ~244MB | Moderate    | Good     | **Default — best balance for most voice notes** |
| medium | ~769MB | Slow        | Better   | When accuracy matters more than speed |
| large  | ~1.5GB | Very slow   | Best     | GPU only — not practical on CPU |

**Why `small`?** It catches most real-world speech errors (accents, background noise, technical terms) without making you wait 2+ minutes per recording on a CPU. The jump from `base` to `small` is the biggest accuracy-per-MB gain in the lineup.

To change it:
```python
# app/transcribe.py
MODEL_SIZE = "base"  # or "tiny", "medium"
```

---

## Project structure

```
voice-notes-summarizer/
├── app/
│   ├── transcribe.py      # faster-whisper wrapper
│   ├── summarize.py       # Groq API summarization
│   └── main.py            # FastAPI endpoint
├── ui/
│   └── streamlit_app.py   # Streamlit frontend
├── eval/
│   ├── fixtures/          # Text fixtures for summarization eval
│   ├── audio_fixtures/    # Audio files + wer_cases.json for WER eval
│   ├── summarization_eval.py
│   ├── wer_eval.py
│   └── run_eval.py
├── recordings/            # Your personal test recordings (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Supported audio formats

mp3, m4a, wav. The FastAPI endpoint validates the MIME type and returns a `415` error for unsupported formats.
