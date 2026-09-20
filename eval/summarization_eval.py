"""
summarization_eval.py — LLM-as-judge evaluation for the summarization step.

For each fixture, it:
  1. Runs summarize_transcript() to get the model's output.
  2. Sends the transcript + model output + ideal output to Groq and asks it
     to score three dimensions on a 1-5 scale.
  3. Returns a list of scored results.

Scoring dimensions:
  - faithfulness:   Is the summary factually accurate to the transcript?
  - completeness:   Does it cover the main points without major omissions?
  - action_recall:  Were all action items correctly identified?
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from project root or eval/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.summarize import get_groq_client, summarize_transcript, GROQ_MODEL

load_dotenv()

FIXTURES_DIR = Path(__file__).parent / "fixtures"

JUDGE_PROMPT = """You are an expert evaluator for AI summarization systems.

You will be given:
- TRANSCRIPT: the original voice note text
- MODEL SUMMARY: the summary produced by the AI
- MODEL ACTION ITEMS: the action items produced by the AI
- IDEAL SUMMARY: a human-written reference summary
- IDEAL ACTION ITEMS: the expected action items

Score the model output on three dimensions, each from 1 to 5:
  faithfulness   — Is the model summary factually accurate to the transcript? (5 = perfectly accurate, 1 = contains hallucinations)
  completeness   — Does the model summary cover the main points? (5 = nothing important missed, 1 = major omissions)
  action_recall  — Were all expected action items identified? (5 = all found, 1 = most missed)

Respond ONLY with valid JSON, exactly in this format:
{
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "action_recall": <1-5>,
  "reasoning": "<one sentence explaining your scores>"
}
"""


def load_fixtures() -> list[dict]:
    """Load all JSON fixture files from the fixtures directory."""
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        with open(path) as f:
            fixtures.append(json.load(f))
    return fixtures


def judge_output(transcript: str, model_output: dict, ideal: dict, client) -> dict:
    """Ask the LLM to score the model's summary against the ideal.

    Args:
        transcript: The original transcript text.
        model_output: Dict with 'summary' and 'action_items' from the model.
        ideal: The fixture dict with 'ideal_summary' and 'ideal_action_items'.
        client: A Groq client instance.

    Returns:
        Dict with faithfulness, completeness, action_recall scores and reasoning.
    """
    user_message = f"""TRANSCRIPT:
{transcript}

MODEL SUMMARY:
{model_output['summary']}

MODEL ACTION ITEMS:
{model_output['action_items']}

IDEAL SUMMARY:
{ideal['ideal_summary']}

IDEAL ACTION ITEMS:
{chr(10).join(f"- {item}" for item in ideal['ideal_action_items'])}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def run_summarization_eval() -> list[dict]:
    """Run the full summarization eval across all fixtures.

    Returns:
        List of result dicts, one per fixture, each containing id, description,
        model output, judge scores, and reasoning.
    """
    client = get_groq_client()
    fixtures = load_fixtures()
    results = []

    for fixture in fixtures:
        model_output = summarize_transcript(fixture["transcript"], client=client)

        if "error" in model_output:
            print(f"  [SKIP] {fixture['id']} — {model_output['error']}")
            continue

        scores = judge_output(fixture["transcript"], model_output, fixture, client)

        results.append({
            "id": fixture["id"],
            "description": fixture["description"],
            "model_summary": model_output["summary"],
            "model_action_items": model_output["action_items"],
            "faithfulness": scores.get("faithfulness"),
            "completeness": scores.get("completeness"),
            "action_recall": scores.get("action_recall"),
            "reasoning": scores.get("reasoning"),
        })

    return results
