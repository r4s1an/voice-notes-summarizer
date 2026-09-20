"""
run_eval.py — Runs both the summarization and transcription evals and prints results.

Usage:
  python eval/run_eval.py              # run both evals
  python eval/run_eval.py --summ-only  # summarization eval only
  python eval/run_eval.py --wer-only   # WER eval only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.summarization_eval import run_summarization_eval
from eval.wer_eval import run_wer_eval


def _divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_summ_results(results: list[dict]) -> None:
    """Print summarization eval results as a readable table."""
    if not results:
        print("  No results.")
        return

    header = f"{'ID':<6} {'Description':<30} {'Faith':>6} {'Comp':>6} {'Recall':>7} {'Avg':>6}"
    print(header)
    print("-" * len(header))

    total_f = total_c = total_r = 0

    for r in results:
        f, c, a = r["faithfulness"], r["completeness"], r["action_recall"]
        avg = round((f + c + a) / 3, 2)
        total_f += f
        total_c += c
        total_r += a
        print(f"{r['id']:<6} {r['description'][:30]:<30} {f:>6} {c:>6} {a:>7} {avg:>6}")
        print(f"       Reasoning: {r['reasoning']}")

    n = len(results)
    avg_f = round(total_f / n, 2)
    avg_c = round(total_c / n, 2)
    avg_r = round(total_r / n, 2)
    overall = round((avg_f + avg_c + avg_r) / 3, 2)
    print("-" * len(header))
    print(f"{'AVG':<6} {'':<30} {avg_f:>6} {avg_c:>6} {avg_r:>7} {overall:>6}")


def _print_wer_results(results: list[dict]) -> None:
    """Print WER eval results as a readable table."""
    if not results:
        return

    header = f"{'Audio':<40} {'WER':>8} {'Rating':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        wer = r["wer"]
        rating = "Excellent" if wer < 0.05 else "Good" if wer < 0.10 else "Fair" if wer < 0.20 else "Poor"
        audio_name = Path(r["audio"]).name
        print(f"{audio_name:<40} {wer:>8.4f} {rating:>10}")
        print(f"  REF: {r['ground_truth'][:80]}")
        print(f"  HYP: {r['hypothesis'][:80]}")

    avg_wer = round(sum(r["wer"] for r in results) / len(results), 4)
    print("-" * len(header))
    print(f"{'AVERAGE WER':<40} {avg_wer:>8.4f}")


def main():
    args = sys.argv[1:]
    run_summ = "--wer-only" not in args
    run_wer = "--summ-only" not in args

    if run_summ:
        _divider("SUMMARIZATION EVAL  (LLM-as-judge, scores 1–5)")
        print("Running... (makes Groq API calls for each fixture + judge call)\n")
        summ_results = run_summarization_eval()
        _print_summ_results(summ_results)

    if run_wer:
        _divider("TRANSCRIPTION EVAL  (Word Error Rate)")
        wer_results = run_wer_eval()
        _print_wer_results(wer_results)

    print()


if __name__ == "__main__":
    main()
