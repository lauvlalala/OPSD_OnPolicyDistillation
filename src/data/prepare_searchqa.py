"""
Prepare SearchQA (Search-R1 style) data for OPSD multi-turn training.

Downloads PeterJinGo/nq_hotpotqa_train from HuggingFace, processes into
verl RL format with Search-R1 system prompt that guides the model to use
<tool_call> for searching.

Usage:
    python -m data.prepare_searchqa --output-dir data/searchqa
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant. "
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def process_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Process a raw dataframe into verl format."""
    records = []
    for idx, row in df.iterrows():
        question = row.get("question", "")
        if not question:
            continue

        # Build prompt
        user_content = SYSTEM_PROMPT.rstrip() + question
        prompt = [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {"role": "user", "content": user_content},
        ]

        # Ground truth
        reward_model = row.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            ground_truth = reward_model["ground_truth"]
        else:
            ground_truth = row.get("golden_answers", [])

        # Wrap ground_truth for EM reward
        if isinstance(ground_truth, list):
            gt_for_reward = {"target": ground_truth}
        else:
            gt_for_reward = {"target": [ground_truth]}

        records.append({
            "data_source": "searchqa",
            "prompt": prompt,
            "ability": "qa",
            "reward_model": {"style": "rule", "ground_truth": gt_for_reward},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
            },
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare SearchQA (Search-R1) for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/searchqa")
    parser.add_argument("--hf-repo", type=str, default="PeterJinGo/nq_hotpotqa_train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        print(f"Downloading {split}.parquet from {args.hf_repo}...")
        local_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=f"{split}.parquet",
            repo_type="dataset",
            local_dir=str(output_dir / "raw"),
        )
        df = pd.read_parquet(local_path)
        print(f"  Loaded {len(df)} rows")

        processed = process_split(df, split)
        out_path = output_dir / f"{split}.parquet"
        processed.to_parquet(out_path)
        print(f"  Saved {len(processed)} samples → {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
