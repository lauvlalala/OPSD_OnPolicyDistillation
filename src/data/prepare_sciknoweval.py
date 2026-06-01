"""
Prepare SciKnowEval for OPSD training.

Downloads hicai-zju/SciKnowEval, filters by domain/level/type, splits into
train/test, and outputs verl RL format.

Usage:
    python -m data.prepare_sciknoweval --output-dir data/sciknoweval --domain Biology
"""

import argparse
import json
from pathlib import Path

import datasets
import pandas as pd


SYSTEM_PROMPT = """Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else."""


def format_choices(choices: dict) -> str:
    """Format choices dict into A: ..., B: ..., etc."""
    texts = choices["text"]
    labels = choices["label"]
    return "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))


def process_sciknoweval(domain: str, levels: list = None, types: list = None, seed: int = 42):
    """Load and process SciKnowEval for a given domain."""
    if levels is None:
        levels = ["L3"]
    if types is None:
        types = ["mcq-4-choices", "mcq-2-choices"]

    print(f"Loading SciKnowEval (domain={domain}, levels={levels})...")
    ds = datasets.load_dataset("hicai-zju/SciKnowEval", split="test")

    # Filter
    ds = ds.filter(lambda x: x["domain"] == domain)
    if levels:
        ds = ds.filter(lambda x: x["details"]["level"] in levels)
    if types:
        ds = ds.filter(lambda x: x["type"] in types)

    print(f"  Filtered: {len(ds)} samples")

    records = []
    for example in ds:
        question = example["question"]
        choices = format_choices(example["choices"])
        answer_key = example["answerKey"]  # "A", "B", "C", or "D"

        prompt_content = f"{question}\n\n{choices}\nPlease reason step by step."

        records.append({
            "data_source": "sciknoweval",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ],
            "ability": "mcq",
            "reward_model": {"style": "mcq", "ground_truth": answer_key},
            "extra_info": {"domain": domain, "question": question},
            "pi_fields": {
                "question": question,
                "choices": choices,
                "answer": answer_key,
            },
        })

    print(f"  Processed {len(records)} samples")
    return records


def split_and_save(records: list, output_dir: Path, train_ratio: float = 0.9, seed: int = 42):
    """Split and save as parquet. Default 90/10 to match SDPO."""
    import random
    random.seed(seed)
    random.shuffle(records)

    split_idx = int(len(records) * train_ratio)
    train_records = records[:split_idx]
    test_records = records[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_records).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(test_records).to_parquet(output_dir / "test.parquet")

    print(f"  Train: {len(train_records)} → {output_dir / 'train.parquet'}")
    print(f"  Test:  {len(test_records)} → {output_dir / 'test.parquet'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SciKnowEval for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/sciknoweval")
    parser.add_argument("--domain", type=str, default="Biology",
                        choices=["Biology", "Chemistry", "Material", "Physics"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = process_sciknoweval(domain=args.domain, seed=args.seed)
    output_dir = Path(args.output_dir) / args.domain.lower()
    split_and_save(records, output_dir, train_ratio=args.train_ratio, seed=args.seed)
    print("Done!")


if __name__ == "__main__":
    main()
