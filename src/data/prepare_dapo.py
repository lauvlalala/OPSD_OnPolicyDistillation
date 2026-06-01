"""
Prepare DAPO-17k for OPSD training.

Downloads BytedTsinghua-SIA/DAPO-Math-17k from HuggingFace, converts to
verl RL format with pi_fields for PI template rendering. Also prepares
MATH-500, AIME 2024, AIME 2025 as eval sets.

Output format (per sample):
  - data_source: "math_dapo"
  - prompt: [{"role": "user", "content": ...}]
  - ability: "math"
  - reward_model: {"style": "rule", "ground_truth": <answer>}
  - pi_fields: {"problem": ..., "answer": ...}

Usage:
    python -m data.prepare_dapo \
        --output-dir data/dapo \
        --train-ratio 0.8
"""

import argparse
import json
import re
from pathlib import Path

import datasets
import pandas as pd


BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# DAPO instruction prefixes to strip
DAPO_INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_INSTRUCTION_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


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


def strip_dapo_instruction(text: str) -> str:
    """Remove DAPO instruction prefix/suffix, add boxed instruction."""
    if text.startswith(DAPO_INSTRUCTION_PREFIX):
        text = text[len(DAPO_INSTRUCTION_PREFIX):]
    if text.endswith(DAPO_INSTRUCTION_SUFFIX):
        text = text[:-len(DAPO_INSTRUCTION_SUFFIX)]
    return text.rstrip() + "\n\n" + BOXED_INSTRUCTION


def process_dapo(train_ratio: float = 0.8, seed: int = 42):
    """Load and process DAPO-17k dataset."""
    print("Loading DAPO-Math-17k from HuggingFace...")
    ds = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    print(f"  Loaded {len(ds)} samples")

    records = []
    skipped = 0

    for example in ds:
        # DAPO has: prompt (str or list), reward_model dict
        if "prompt" in example and isinstance(example["prompt"], list):
            # Already chat format
            messages = example["prompt"]
            problem = ""
            for msg in messages:
                if msg["role"] == "user":
                    problem = strip_dapo_instruction(msg["content"])
                    msg["content"] = problem
        elif "prompt" in example:
            problem = strip_dapo_instruction(str(example["prompt"]))
            messages = [{"role": "user", "content": problem}]
        else:
            skipped += 1
            continue

        # Get ground truth answer
        reward_model = example.get("reward_model", {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except (json.JSONDecodeError, TypeError):
                reward_model = {}

        ground_truth = reward_model.get("ground_truth", None)
        if ground_truth is None:
            skipped += 1
            continue

        records.append({
            "data_source": "math_dapo",
            "prompt": messages,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
            "pi_fields": {
                "problem": problem,
                "answer": str(ground_truth),
            },
        })

    if skipped:
        print(f"  Skipped {skipped} samples")
    print(f"  Processed {len(records)} valid samples")

    # Split
    ds_hf = datasets.Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(ds_hf) * train_ratio)
    train_records = [ds_hf[i] for i in range(split_idx)]
    val_records = [ds_hf[i] for i in range(split_idx, len(ds_hf))]

    return train_records, val_records


def process_math500(data_path: Path = None):
    """Process MATH-500 eval set."""
    if data_path and data_path.exists():
        print(f"Loading MATH-500 from {data_path}...")
        data_list = []
        with open(data_path) as f:
            for line in f:
                data_list.append(json.loads(line))
    else:
        print("Loading MATH-500 from HuggingFace...")
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        data_list = list(ds)

    records = []
    for example in data_list:
        problem = example.get("problem", "")
        answer = example.get("answer", "")
        if not problem or not answer:
            continue
        prompt_content = problem.rstrip() + "\n\n" + BOXED_INSTRUCTION
        records.append({
            "data_source": "math500",
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "pi_fields": {"problem": problem, "answer": str(answer)},
        })
    print(f"  MATH-500: {len(records)} problems")
    return records


def save_parquet(records: list, path: Path):
    """Save records as parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path)
    print(f"  Saved: {path} ({len(records)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Prepare DAPO-17k for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/dapo")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--math500-path", type=str, default=None, help="Local path to MATH-500 test.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Process DAPO training data
    train_records, val_records = process_dapo(train_ratio=args.train_ratio, seed=args.seed)
    save_parquet(train_records, output_dir / "train.parquet")
    save_parquet(val_records, output_dir / "val_dapo.parquet")

    # Process eval sets
    math500_path = Path(args.math500_path) if args.math500_path else None
    math500_records = process_math500(math500_path)
    save_parquet(math500_records, output_dir / "val_math500.parquet")

    print("Done!")


if __name__ == "__main__":
    main()
