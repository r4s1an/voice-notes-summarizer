"""
summarize.py — LLM summarization via Groq API.

Sends a transcript to Groq and returns a structured response with:
  - A concise summary of what was said
  - A bullet-point list of action items
"""

import os
import sys
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# The Groq model to use for summarization.
# llama-3.3-70b-versatile is fast, cheap, and handles summarization well.
GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a helpful assistant that processes voice note transcripts.
Given a transcript, you will return:
1. A concise summary (2-4 sentences)
2. A numbered list of action items extracted from the transcript

Format your response exactly like this:
SUMMARY:
<your summary here>

ACTION ITEMS:
1. <action item>
2. <action item>
...

If there are no clear action items, write "No action items identified."
"""


def get_groq_client() -> Groq:
    """Create and return a Groq API client using GROQ_API_KEY from the environment.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def summarize_transcript(transcript: str, client: Groq | None = None) -> dict:
    """Send a transcript to Groq and return a summary and action items.

    Args:
        transcript: The full transcript text to summarize.
        client: Optional pre-built Groq client. If None, one is created fresh.

    Returns:
        A dict with keys "summary" (str) and "action_items" (str).
        On rate-limit error, returns a dict with key "error" (str).
    """
    if client is None:
        client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
            temperature=0.3,  # low temperature = more focused, less creative
        )

        raw = response.choices[0].message.content
        return _parse_response(raw)

    except RateLimitError:
        return {
            "error": "Groq rate limit reached. Please wait a moment and try again."
        }


def _parse_response(raw: str) -> dict:
    """Split the raw LLM response into summary and action_items sections.

    Args:
        raw: The full text response from Groq.

    Returns:
        Dict with "summary" and "action_items" keys.
    """
    summary = ""
    action_items = ""

    if "SUMMARY:" in raw and "ACTION ITEMS:" in raw:
        parts = raw.split("ACTION ITEMS:")
        summary = parts[0].replace("SUMMARY:", "").strip()
        action_items = parts[1].strip()
    else:
        # Fallback: return everything as summary if format is unexpected
        summary = raw.strip()
        action_items = "Could not parse action items."

    return {"summary": summary, "action_items": action_items}


# --- Standalone test ---
SAMPLE_TRANSCRIPT = (
    "This is a test of the voice notes summarizer. "
    "Today I need to review the project, send emails, and finish the report. "
    "Also, I should schedule a meeting with the team for Thursday and follow up "
    "with the client about the proposal we sent last week."
)

if __name__ == "__main__":
    print("Calling Groq API...")
    result = summarize_transcript(SAMPLE_TRANSCRIPT)

    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    print("\n--- SUMMARY ---")
    print(result["summary"])
    print("\n--- ACTION ITEMS ---")
    print(result["action_items"])
