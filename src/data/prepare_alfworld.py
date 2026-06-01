"""
Prepare ALFWorld data for OPSD multi-turn training.

Generates initial task prompt parquets for ALFWorld text environment.
The actual environment interaction happens at runtime via the AlfWorldTool.

Usage:
    python -m data.prepare_alfworld --output-dir data/alfworld --data-size 128
"""

import argparse
import os
from pathlib import Path

import datasets
import pandas as pd


def make_record(idx: int, split: str) -> dict:
    """Create a minimal verl record for ALFWorld.

    The prompt is empty because ALFWorld tasks are assigned by the environment
    at runtime (env.reset() returns the task description as initial observation).
    """
    return {
        "data_source": "alfworld",
        "prompt": [{"role": "user", "content": ""}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": split, "index": idx},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ALFWorld for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/alfworld")
    parser.add_argument("--train-data-size", type=int, default=128)
    parser.add_argument("--val-data-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate placeholder parquets (env provides actual tasks at runtime)
    train_records = [make_record(i, "train") for i in range(args.train_data_size)]
    test_records = [make_record(i, "test") for i in range(args.val_data_size)]

    pd.DataFrame(train_records).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(test_records).to_parquet(output_dir / "test.parquet")

    print(f"  Train: {len(train_records)} → {output_dir / 'train.parquet'}")
    print(f"  Test:  {len(test_records)} → {output_dir / 'test.parquet'}")
    print("Done! (Note: actual task prompts come from ALFWorld env at runtime)")


if __name__ == "__main__":
    main()
