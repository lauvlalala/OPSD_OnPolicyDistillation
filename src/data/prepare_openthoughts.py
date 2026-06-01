"""
Prepare OpenThoughts-30k for OPSD training.

Downloads a 30k subset from open-thoughts/OpenThoughts-114k on HuggingFace,
converts to verl RL format with pi_fields for PI template rendering.

Output format (per sample):
  - data_source: "openthoughts"
  - prompt: [{"role": "user", "content": ...}]
  - ability: "math"
  - reward_model: {"style": "rule", "ground_truth": <final_answer>}
  - pi_fields: {"problem": ..., "solution": ..., "answer": ...}

Usage:
    python -m data.prepare_openthoughts \
        --output-dir data/openthoughts \
        --num-samples 30000 \
        --train-ratio 0.9
"""

import argparse
import re
from pathlib import Path

import datasets
import pandas as pd


BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} from text."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[idx + len("\\boxed{"):i - 1].strip()
    return None


def process_openthoughts(num_samples: int = 30000, seed: int = 42):
    """Load and process OpenThoughts dataset."""
    print(f"Loading OpenThoughts-114k from HuggingFace (taking {num_samples} samples)...")
    ds = datasets.load_dataset("open-thoughts/OpenThoughts-114k", split="train")

    # Shuffle and take subset
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    print(f"  Selected {len(ds)} samples")

    records = []
    skipped = 0

    for example in ds:
        # OpenThoughts has: system, conversations (list of user/assistant turns)
        conversations = example.get("conversations", [])
        if not conversations:
            skipped += 1
            continue

        # Extract problem (first user message) and solution (first assistant response)
        problem = None
        solution = None
        for turn in conversations:
            if turn.get("from") == "human" or turn.get("role") == "user":
                if problem is None:
                    problem = turn.get("value") or turn.get("content", "")
            elif turn.get("from") == "gpt" or turn.get("role") == "assistant":
                if solution is None:
                    solution = turn.get("value") or turn.get("content", "")

        if not problem or not solution:
            skipped += 1
            continue

        # Extract final answer from solution
        answer = extract_boxed_answer(solution)
        if answer is None:
            # Try to get from metadata
            answer = example.get("answer", None)
        if answer is None:
            skipped += 1
            continue

        # Build verl format
        prompt_content = problem.rstrip()
        if "\\boxed" not in prompt_content:
            prompt_content += "\n\n" + BOXED_INSTRUCTION

        records.append({
            "data_source": "openthoughts",
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "pi_fields": {
                "problem": problem,
                "solution": solution,
                "answer": answer,
            },
        })

    if skipped:
        print(f"  Skipped {skipped} samples (missing problem/solution/answer)")
    print(f"  Processed {len(records)} valid samples")
    return records


def split_and_save(records: list, output_dir: Path, seed: int = 42):
    """Save all records as training data (no split — use external benchmarks for eval)."""
    import random
    random.seed(seed)
    random.shuffle(records)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    pd.DataFrame(records).to_parquet(train_path)

    print(f"  Train: {len(records)} → {train_path}")
    print(f"  (Validation uses external benchmarks: MATH-500, AIME 2024/2025)")


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenThoughts-30k for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/openthoughts")
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = process_openthoughts(num_samples=args.num_samples, seed=args.seed)
    split_and_save(records, Path(args.output_dir), seed=args.seed)
    print("Done!")


if __name__ == "__main__":
    main()
