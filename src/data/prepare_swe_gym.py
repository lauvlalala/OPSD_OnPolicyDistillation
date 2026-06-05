"""
Prepare SWE-Gym Lite for OPSD multi-turn training.

Downloads SWE-Gym/SWE-Gym-Lite from HuggingFace and converts to verl format.
Each instance includes the docker image name and test command for runtime use.

Usage:
    python -m data.prepare_swe_gym --output-dir data/swe_gym
"""

import argparse
from pathlib import Path

import datasets
import pandas as pd


SWE_SYSTEM_PROMPT = (
    "You are an expert software engineer. You will be given a GitHub issue "
    "and access to a bash shell in the repository. Your goal is to fix the issue "
    "by modifying the code. Use bash commands to:\n"
    "1. Explore the repository structure\n"
    "2. Read relevant files\n"
    "3. Understand the bug\n"
    "4. Edit the code to fix it\n"
    "5. Run tests to verify your fix\n\n"
    "When you believe the issue is fixed, run the test suite to confirm."
)

DOCKER_IMAGE_PREFIX = "xingyaoww/sweb.eval.x86_64."


def instance_id_to_docker_image(instance_id: str) -> str:
    """Convert instance_id to Docker image name (lowercase, __ -> _s_)."""
    return f"{DOCKER_IMAGE_PREFIX}{instance_id.replace('__', '_s_').lower()}"


def process_swe_gym_lite(seed: int = 42):
    """Load and process SWE-Gym Lite dataset."""
    print("Loading SWE-Gym-Lite from HuggingFace...")
    ds = datasets.load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
    print(f"  Loaded {len(ds)} instances")

    records = []
    for example in ds:
        instance_id = example.get("instance_id", "")
        problem_statement = example.get("problem_statement", "")
        repo = example.get("repo", "")
        test_patch = example.get("test_patch", "")

        if not instance_id or not problem_statement:
            continue

        docker_image = instance_id_to_docker_image(instance_id)

        # Build test command from test_patch or use default
        test_cmd = example.get("test_cmd", "")
        if not test_cmd:
            # Fallback: run pytest on the test files mentioned in test_patch
            test_cmd = "python -m pytest --tb=short -q"

        # Build prompt
        prompt_content = f"{SWE_SYSTEM_PROMPT}\n\n## Issue\n\n{problem_statement}"

        records.append({
            "data_source": "swe_gym",
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "instance_id": instance_id,
                "docker_image": docker_image,
                "test_cmd": test_cmd,
                "repo": repo,
                "issue_text": problem_statement,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "bash": {
                        "create_kwargs": {
                            "docker_image": docker_image,
                            "test_cmd": test_cmd,
                            "issue_text": problem_statement,
                        }
                    }
                },
            },
        })

    print(f"  Processed {len(records)} valid instances")
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare SWE-Gym Lite for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/swe_gym")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = process_swe_gym_lite(seed=args.seed)

    # Use all 230 instances for both train and test (same as SDAR pattern)
    pd.DataFrame(records).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(records).to_parquet(output_dir / "test.parquet")

    print(f"  Saved: {output_dir / 'train.parquet'} ({len(records)} instances)")
    print(f"  Saved: {output_dir / 'test.parquet'} ({len(records)} instances)")
    print("Done!")


if __name__ == "__main__":
    main()
