"""
wer_eval.py — Word Error Rate evaluation for the transcription step.

Compares faster-whisper output against known ground-truth transcripts.

To add test cases, create a file at eval/audio_fixtures/wer_cases.json:
[
  {
    "audio": "eval/audio_fixtures/sample1.m4a",
    "ground_truth": "The exact words spoken in the recording."
  },
  ...
]

WER formula: (substitutions + deletions + insertions) / total reference words.
  0.0 = perfect,  1.0 = completely wrong.
  A score below 0.10 (10%) is considered very good for clear speech.
"""

import json
import sys
from pathlib import Path

from jiwer import wer as compute_wer

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.transcribe import load_model, transcribe_audio

WER_CASES_PATH = Path(__file__).parent / "audio_fixtures" / "wer_cases.json"


def run_wer_eval() -> list[dict]:
    """Run WER evaluation on all audio fixtures defined in wer_cases.json.

    Returns:
        List of result dicts with keys: audio, wer, ground_truth, hypothesis.
        Returns an empty list with a warning if no cases file is found.
    """
    if not WER_CASES_PATH.exists():
        print(
            f"  [SKIP] No WER test cases found at {WER_CASES_PATH}\n"
            "  To add cases, create that file — see the docstring at the top of wer_eval.py."
        )
        return []

    with open(WER_CASES_PATH) as f:
        cases = json.load(f)

    model = load_model()
    results = []

    for case in cases:
        audio_path = case["audio"]
        ground_truth = case["ground_truth"].lower().strip()

        hypothesis = transcribe_audio(audio_path, model=model).lower().strip()
        score = compute_wer(ground_truth, hypothesis)

        results.append({
            "audio": audio_path,
            "wer": round(score, 4),
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
        })

    return results
