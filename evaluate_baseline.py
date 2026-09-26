"""Zero-shot baseline for a model on the test split, run before fine-tuning."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Reuse Ollama helpers from the main app
try:
    import requests
except ImportError:
    subprocess = __import__("subprocess")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import requests

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"


def ollama_generate(model, prompt, system=None, temperature=0.3, timeout=120):
    """Simple Ollama generation."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9},
    }
    if system:
        payload["system"] = system
    try:
        r = requests.post(OLLAMA_GENERATE, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def load_dataset(path, fmt="sharegpt"):
    """Load dataset in ShareGPT or Alpaca format."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip metadata header
            if "metadata" in rec and len(rec) == 1:
                continue
            records.append(rec)
    return records


def extract_prompt_response(record, fmt="sharegpt"):
    """Extract (prompt, response, task_type) from a record."""
    if fmt == "sharegpt":
        msgs = record.get("messages", [])
        prompt = ""
        reference = ""
        for m in msgs:
            if m.get("role") == "user":
                prompt = m.get("content", "")
            elif m.get("role") == "assistant":
                reference = m.get("content", "")
        meta = record.get("metadata", {})
        return prompt, reference, meta.get("task", "unknown")

    elif fmt == "alpaca":
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
        reference = record.get("output", "")
        meta = record.get("metadata", {})
        return prompt, reference, meta.get("task", "unknown")

    return "", "", "unknown"


def evaluate_response(prediction, reference, task):
    """Compute evaluation metrics for a single prediction."""
    metrics = {}

    # 1. Response length ratio (pred vs reference)
    ref_len = len(reference.split())
    pred_len = len(prediction.split())
    if ref_len > 0:
        ratio = pred_len / ref_len
        # Ideal ratio is close to 1.0, penalize being way off
        metrics["length_ratio"] = round(min(ratio, 2.0) / 2.0, 3)  # 0.5 is "perfect" -> 0.25 score, cap at 1.0
        metrics["length_balance"] = round(1.0 - min(abs(ratio - 1.0), 1.0), 3)
    else:
        metrics["length_ratio"] = 0.0
        metrics["length_balance"] = 0.0

    # 2. Word overlap (simple proxy for content coverage)
    ref_words = set(reference.lower().split())
    pred_words = set(prediction.lower().split())
    if ref_words:
        overlap = len(ref_words & pred_words) / len(ref_words)
        metrics["word_overlap"] = round(overlap, 3)
    else:
        metrics["word_overlap"] = 0.0

    # 3. Format adherence (task-specific checks)
    format_score = 0.0
    if task in ("step_by_step",):
        # Should have numbered list
        if any(re.search(r"^\d+\.", line) for line in prediction.split("\n")):
            format_score = 1.0
    elif task in ("explanation", "compare", "critique"):
        # Should have some structure (headers or paragraphs)
        if "#" in prediction or len(prediction.split("\n\n")) >= 2:
            format_score = 1.0
    elif task in ("summary", "application", "qa"):
        # Should be plain text paragraph(s), no markdown
        if "#" not in prediction and "*" not in prediction:
            format_score = 1.0
    elif task in ("creative_rewrite",):
        # Should match requested format (hard to check automatically)
        format_score = 0.5  # neutral
    else:
        format_score = 0.5
    metrics["format_adherence"] = round(format_score, 2)

    # 4. Response completeness (has substantial content)
    metrics["has_content"] = 1.0 if pred_len >= 10 else 0.0

    # 5. Composite score (weighted average)
    weights = {
        "length_balance": 0.2,
        "word_overlap": 0.35,
        "format_adherence": 0.3,
        "has_content": 0.15,
    }
    composite = sum(metrics.get(k, 0) * w for k, w in weights.items())
    metrics["composite"] = round(composite, 3)

    return metrics


def run_baseline_eval(args):
    print(f"Loading dataset: {args.dataset}")
    records = load_dataset(args.dataset, args.format)
    print(f"Loaded {len(records)} records")

    # Sample if needed
    if args.max_samples and len(records) > args.max_samples:
        import random
        random.seed(42)
        records = random.sample(records, args.max_samples)
        print(f"Sampled {len(records)} records for evaluation")

    # Group by task
    by_task = defaultdict(list)
    for rec in records:
        prompt, reference, task = extract_prompt_response(rec, args.format)
        if prompt and reference:
            by_task[task].append((prompt, reference))

    print(f"Task distribution: {dict((t, len(v)) for t, v in by_task.items())}")

    # Evaluate
    results = []
    task_scores = defaultdict(list)

    for task, items in by_task.items():
        print(f"\nEvaluating task: {task} ({len(items)} items)")
        for idx, (prompt, reference) in enumerate(items):
            if args.verbose:
                print(f"  [{idx+1}/{len(items)}] {prompt[:80]}...")

            # Generate prediction
            pred = ollama_generate(
                args.model,
                prompt,
                system=f"You are a helpful assistant specializing in {args.domain}.",
                temperature=0.3,
            )

            # Evaluate
            metrics = evaluate_response(pred, reference, task)
            metrics["task"] = task
            metrics["prompt_preview"] = prompt[:100]
            metrics["prediction_preview"] = pred[:150]

            results.append(metrics)
            task_scores[task].append(metrics["composite"])

            if args.verbose:
                print(f"    -> composite: {metrics['composite']}")

            # Small delay to not hammer Ollama
            time.sleep(0.5)

    # Aggregate report
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "domain": args.domain,
        "total_evaluated": len(results),
        "overall_average": round(sum(r["composite"] for r in results) / len(results), 3) if results else 0,
        "per_task": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Scores are 0-1. Higher is better. This is your pre-training baseline.",
    }

    for task, scores in task_scores.items():
        report["per_task"][task] = {
            "count": len(scores),
            "average": round(sum(scores) / len(scores), 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
        }

    # Save report
    report_path = args.output or os.path.join(
        os.path.dirname(args.dataset),
        f"baseline_report_{args.model.replace(':', '_')}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("BASELINE EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Overall average score: {report['overall_average']}")
    print(f"\nPer-task breakdown:")
    for task, stats in report["per_task"].items():
        print(f"  {task:20s}: avg={stats['average']:.3f}  (n={stats['count']})")
    print(f"\nReport saved to: {report_path}")
    print(f"\nTip: After fine-tuning, run this again with the same dataset.")
    print(f"      If your fine-tuned model scores higher, the training worked!")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline evaluation for instruction-tuning datasets"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset (.jsonl)")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., qwen3:8b)")
    parser.add_argument("--format", choices=["sharegpt", "alpaca"], default="sharegpt",
                        help="Dataset format")
    parser.add_argument("--domain", default="Design", help="Domain/topic")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples to evaluate (default: 50)")
    parser.add_argument("--output", help="Output report path (auto-generated if omitted)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    run_baseline_eval(args)


if __name__ == "__main__":
    main()
