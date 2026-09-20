# Technical Reference — Voice Notes Summarizer

A complete top-to-bottom explanation of how the project is built, why each decision was made, and how every piece fits together.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Data flow](#2-data-flow)
3. [app/transcribe.py](#3-apptranscribepy)
4. [app/summarize.py](#4-appsummarizepy)
5. [app/main.py](#5-appmainpy)
6. [ui/streamlit_app.py](#6-uistreamlit_apppy)
7. [Eval harness](#7-eval-harness)
8. [Configuration and secrets](#8-configuration-and-secrets)
9. [Dependencies explained](#9-dependencies-explained)
10. [Known limitations and failure modes](#10-known-limitations-and-failure-modes)

---

## 1. Architecture overview

```
┌──────────────────┐         HTTP multipart/form-data
│  Streamlit UI    │  ────────────────────────────────►  ┌──────────────────┐
│  (port 8501)     │                                      │  FastAPI backend │
│                  │  ◄────────────────────────────────   │  (port 8000)     │
└──────────────────┘         JSON response                └────────┬─────────┘
                                                                   │
                                                    ┌──────────────┴──────────────┐
                                                    │                             │
                                             ┌──────▼──────┐             ┌───────▼───────┐
                                             │   Whisper   │             │   Groq API    │
                                             │  (local CPU)│             │  (cloud LLM)  │
                                             └─────────────┘             └───────────────┘
```

The system has two processes that run concurrently:

- **FastAPI** handles all heavy work: receiving the file, running transcription, calling Groq, and returning structured JSON.
- **Streamlit** is a pure frontend: it accepts user input, sends one HTTP request to FastAPI, and renders the response. It has no business logic of its own.

This separation means you can replace the Streamlit UI with anything (a CLI, a mobile app, a different web framework) without touching the core logic.

---

## 2. Data flow

```
User uploads audio file
        │
        ▼
Streamlit sends POST /process (multipart form, binary audio bytes)
        │
        ▼
FastAPI receives UploadFile
        │
        ├─► Validate MIME type (415 if unsupported)
        │
        ├─► Write bytes to a temp file on disk
        │       (faster-whisper needs a file path, not a byte stream)
        │
        ├─► transcribe_audio(tmp_path) → transcript string
        │       Whisper reads file, runs CTC beam search, yields segments
        │       Segments are joined into one plain-text string
        │
        ├─► Delete temp file
        │
        ├─► summarize_transcript(transcript) → {summary, action_items}
        │       Sends transcript to Groq chat completions API
        │       Response is parsed into two named sections
        │
        └─► Return JSON: {transcript, summary, action_items}
                │
                ▼
        Streamlit renders three panels: transcript, summary, action items
```

---

## 3. app/transcribe.py

### Purpose
Wraps the `faster-whisper` library so the rest of the project only needs to call one function.

### Key design decisions

**Why faster-whisper instead of OpenAI's whisper package?**
faster-whisper re-implements Whisper using CTranslate2, a C++ inference engine. On CPU it is 2–4× faster than the original and uses less memory because it runs `int8` quantized weights by default. The API is nearly identical.

**Why load the model once at startup?**
In `main.py`, `load_model()` is called at module import time (outside any request handler). Loading a ~244MB model takes 2–5 seconds. If it were loaded per request, every upload would have that cold-start cost. Loading once at startup means the model stays in RAM and every request reuses it.

**Why `beam_size=5`?**
Beam search explores multiple possible transcription paths simultaneously and picks the highest-probability one. `beam_size=5` is the standard default — it balances accuracy against compute. `beam_size=1` is greedy decoding (fastest, less accurate). Values above 5 give diminishing returns on CPU.

**The `initial_prompt` parameter**
Whisper was trained on internet text, so it handles everyday English well but guesses on proper nouns and technical jargon. Passing an `initial_prompt` string (e.g. `"backend, API, deployment"`) gives Whisper a context window before the audio starts — it biases the token probabilities toward those words. This is how you fix recurring mishearing errors without switching to a larger model.

### Function signatures

```python
load_model(model_size: str = "small") -> WhisperModel
transcribe_audio(file_path: str, model=None, initial_prompt=None) -> str
```

---

## 4. app/summarize.py

### Purpose
Calls the Groq API with a structured prompt and parses the response into `summary` and `action_items`.

### Key design decisions

**Why Groq instead of running an LLM locally?**
Local LLMs large enough to summarize well (7B+ parameters) are slow on CPU and require significant RAM. Groq runs inference on custom LPU hardware and returns responses in under a second. For summarization (where the output is short), the API cost is negligible.

**System prompt design**
The system prompt instructs the model to return output in a strict format:
```
SUMMARY:
<text>

ACTION ITEMS:
1. <item>
```
This makes `_parse_response()` simple and reliable — it splits on the `ACTION ITEMS:` marker. A free-form prompt would require more complex parsing.

**Why `temperature=0.3`?**
Temperature controls randomness. At `0`, the model always picks the highest-probability next token (deterministic but sometimes repetitive). At `1`, it samples freely (creative but inconsistent). `0.3` keeps the output focused and factual, which is what you want for a summary, while avoiding the exact same phrasing every time.

**Rate limit handling**
The Groq SDK raises `groq.RateLimitError` when the account hits its requests-per-minute or tokens-per-minute limit. We catch it and return a friendly dict with an `"error"` key instead of raising an exception. The FastAPI layer translates that into a `429` HTTP response; the Streamlit layer shows it as a `st.warning()` banner.

### Function signatures

```python
get_groq_client() -> Groq
summarize_transcript(transcript: str, client=None) -> dict   # {summary, action_items} or {error}
_parse_response(raw: str) -> dict
```

---

## 5. app/main.py

### Purpose
FastAPI application with two endpoints: `GET /health` and `POST /process`.

### Startup behavior
Two expensive objects are created at module import time and reused across all requests:
```python
whisper_model = load_model()   # ~2-5s, loads model weights into RAM
groq_client = get_groq_client()  # validates API key, creates HTTP session
```
FastAPI's `--reload` flag (used in development) restarts the worker process when files change, so both objects are re-initialized on each reload.

### The /process endpoint

**File upload with `UploadFile`**
FastAPI's `UploadFile` streams the file into memory. We call `await file.read()` to get the bytes, then write them to a `tempfile.NamedTemporaryFile`. The temp file is deleted after transcription whether or not it succeeds.

**MIME type validation**
Browsers and curl send a `Content-Type` header with the file. We check it against an allowlist. The full set we accept:
```
audio/mpeg      — mp3
audio/mp4       — m4a (some browsers)
audio/m4a       — m4a (other browsers)
audio/x-m4a     — m4a (Windows Voice Recorder specifically)
audio/wav       — wav
audio/x-wav     — wav (alternate MIME)
```
We discovered `audio/x-m4a` was missing during testing — Windows Voice Recorder uses it.

**The `initial_prompt` form field**
FastAPI automatically maps form fields alongside `File(...)` uploads. The Streamlit UI passes it as a `data={"initial_prompt": ...}` field in the same multipart request.

**Error hierarchy**
| Scenario | HTTP status | Source |
|---|---|---|
| Unsupported file type | 415 | Validation in endpoint |
| Transcription crash | 500 | Exception from Whisper |
| Groq rate limit | 429 | `"error"` key in summarize result |

---

## 6. ui/streamlit_app.py

### Purpose
Browser-based UI. Entirely stateless — every "Process" click is a fresh HTTP request to FastAPI.

### Layout
```
[file uploader]
[audio preview player]
[domain hint text input]  ← optional initial_prompt
[Process button]

[Transcript column]  |  [Summary column]
                     |  [Action Items column]
```

Two-column layout is achieved with `st.columns(2)`. The transcript goes in a `st.text_area` (scrollable, selectable) and the summary/action items go in `st.write` (rendered as markdown).

### Error handling in the UI
Three distinct failure states are shown differently:
- **Rate limit (429)** → `st.warning()` — yellow banner, not user's fault
- **Other API error** → `st.error()` — red banner with the detail message
- **Backend unreachable** → `st.error()` with a specific message telling the user to check FastAPI is running
- **Timeout** → `st.error()` suggesting the audio may be too long

---

## 7. Eval harness

The eval harness lives in `eval/` and measures quality in two independent dimensions.

### Summarization eval — LLM-as-judge

**How it works**
For each fixture in `eval/fixtures/`, it:
1. Calls `summarize_transcript()` to get the model's actual output
2. Sends the transcript + model output + ideal output to Groq with a judge prompt
3. Groq returns JSON scores (1–5) on three dimensions

**Why LLM-as-judge?**
Traditional NLP metrics like ROUGE measure n-gram overlap between the generated and reference text. A good paraphrase can score poorly because it uses different words, and a bad summary that copies phrases verbatim can score well. An LLM judge understands meaning, not just token overlap, so it better reflects what we actually care about.

**Scoring dimensions**

| Dimension | Question |
|---|---|
| faithfulness | Does the summary contain only facts from the transcript? (detects hallucination) |
| completeness | Does it cover all the main points? (detects omissions) |
| action_recall | Were all action items found? (task-specific correctness) |

**The judge uses `response_format={"type": "json_object"}`**
This is Groq's JSON mode — it guarantees the response is valid JSON, eliminating the need to parse free-form text.

### WER eval — transcription quality

**Word Error Rate formula**
```
WER = (Substitutions + Deletions + Insertions) / Total words in reference
```
Computed by the `jiwer` library using dynamic programming (same algorithm as edit distance). Both strings are lowercased before comparison so capitalization doesn't affect the score.

**Rating thresholds used in the output**
| WER | Rating |
|---|---|
| < 0.05 | Excellent |
| 0.05–0.10 | Good |
| 0.10–0.20 | Fair |
| > 0.20 | Poor |

**Adding audio test cases**
Create `eval/audio_fixtures/wer_cases.json`:
```json
[
  {"audio": "eval/audio_fixtures/001.m4a", "ground_truth": "exact words spoken"}
]
```
The `ground_truth` should be what was actually said, not what you intended to say — the point is to measure the model's accuracy against reality.

---

## 8. Configuration and secrets

All secrets are loaded from `.env` via `python-dotenv`. The `load_dotenv()` call appears at the top of every module that needs environment variables.

**Why not load it only in `main.py`?**
Each module (`summarize.py`, `transcribe.py`) can also be run as a standalone script. If `load_dotenv()` were only in `main.py`, running `python app/summarize.py` directly would fail to find `GROQ_API_KEY`. Calling `load_dotenv()` in each module is safe — subsequent calls are no-ops if the env vars are already set.

**Variables**

| Variable | Used by | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `summarize.py` | Authenticates Groq API requests |
| `HF_TOKEN` | faster-whisper / HuggingFace Hub | Increases model download rate limits (optional — public models work without it) |

---

## 9. Dependencies explained

| Package | Why it's here |
|---|---|
| `faster-whisper` | Local speech-to-text. Faster than OpenAI's whisper package on CPU via CTranslate2. |
| `groq` | Official Groq Python SDK. Handles auth, retries, and typed responses. |
| `fastapi` | Async web framework. Native support for file uploads, form fields, and JSON responses. |
| `uvicorn[standard]` | ASGI server that runs FastAPI. The `[standard]` extra adds WebSocket support and faster event loop. |
| `streamlit` | Turns a Python script into a web UI with minimal boilerplate. |
| `pydub` | Audio format conversion utility. Sits alongside ffmpeg for cases where format normalization is needed. |
| `python-dotenv` | Loads `.env` files into `os.environ` at runtime. |
| `python-multipart` | Required by FastAPI to parse `multipart/form-data` requests (file uploads). Without it, `UploadFile` silently fails. |
| `requests` | HTTP client used by Streamlit to call the FastAPI backend. |
| `jiwer` | Computes Word Error Rate for the transcription eval. |

---

## 10. Known limitations and failure modes

**Long audio files**
Whisper processes audio in 30-second chunks internally. Very long recordings (30+ minutes) will work but may take several minutes on CPU. The Streamlit timeout is set to 120 seconds — increase it in `streamlit_app.py` if needed.

**Technical jargon**
Whisper was trained on general internet text. Uncommon proper nouns, acronyms, and domain-specific terms (e.g. "Kubernetes", "OIDC", company names) are frequently mis-transcribed. The `initial_prompt` parameter is the primary mitigation.

**Groq model availability**
Available models vary by account tier and change over time. If you get a `404 model_not_found` error, run `python list_models.py` and update `GROQ_MODEL` in `app/summarize.py`.

**Audio with multiple speakers**
Whisper does not do speaker diarization (labeling who said what). All speech is merged into a single transcript. The summary will reflect this — action items won't have speaker attribution unless the speakers identify themselves by name.

**Streamlit + FastAPI on the same machine**
Both servers must be running simultaneously. If FastAPI is stopped, Streamlit will show a `ConnectionError` banner. There is no automatic restart — you must relaunch `uvicorn` manually.
