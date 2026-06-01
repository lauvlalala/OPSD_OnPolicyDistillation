"""
Prepare ToolUse data for OPSD training.

Processes the SDPO tooluse format (JSON with prompt/answer/kind fields)
into verl RL format.

The data can be obtained from the SDPO repository:
    datasets/tooluse/train.json
    datasets/tooluse/test.json

Usage:
    python -m data.prepare_tooluse --data-dir /path/to/tooluse --output-dir data/tooluse
"""

import argparse
import json
from pathlib import Path

import datasets
import pandas as pd


def process_tooluse(data_path: Path, split: str):
    """Load and process tooluse JSON file."""
    json_file = data_path / f"{split}.json"
    print(f"Loading {json_file}...")
    ds = datasets.load_dataset("json", data_files=str(json_file), split="train")
    print(f"  Loaded {len(ds)} samples")

    records = []
    skipped = 0

    for example in ds:
        prompt = example.get("prompt", "")
        answer = example.get("answer", "")
        system = example.get("system", None)

        if not prompt or not answer:
            skipped += 1
            continue

        # Build chat messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # answer is JSON string of action list: [{"Action": ..., "Action_Input": ...}]
        ground_truth = answer if isinstance(answer, str) else json.dumps(answer)

        records.append({
            "data_source": "tooluse",
            "prompt": messages,
            "ability": "tooluse",
            "reward_model": {"style": "tooluse", "ground_truth": ground_truth},
            "extra_info": {"split": split, "description": example.get("description", "")},
            "pi_fields": {
                "prompt": prompt,
                "answer": ground_truth,
            },
        })

    if skipped:
        print(f"  Skipped {skipped} samples")
    print(f"  Processed {len(records)} samples")
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare ToolUse for OPSD")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.json and test.json")
    parser.add_argument("--output-dir", type=str, default="data/tooluse")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        json_file = data_path / f"{split}.json"
        if not json_file.exists():
            print(f"  Warning: {json_file} not found, skipping")
            continue
        records = process_tooluse(data_path, split)
        pd.DataFrame(records).to_parquet(output_dir / f"{split}.parquet")
        print(f"  Saved: {output_dir / f'{split}.parquet'}")

    print("Done!")


if __name__ == "__main__":
    main()
