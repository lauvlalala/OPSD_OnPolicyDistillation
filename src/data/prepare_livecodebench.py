"""
Prepare LiveCodeBench v6 for OPSD training.

Matches SDPO's approach:
- Time-based split: problems before cutoff date → train, after → test
- Train set uses 50% of test cases (reduced) for reward signal
- Test set uses full test cases for evaluation

Usage:
    python -m data.prepare_livecodebench --output-dir data/livecodebench
"""

import argparse
import base64
import copy
import json
import pickle
import zlib
from datetime import datetime
from pathlib import Path

import datasets
import numpy as np
import pandas as pd


TRAIN_CUTOFF = datetime(2025, 2, 1)
TEST_CASES_KEEP_RATIO = 0.5
TIME_LIMIT = 6

CODE_PROMPT = (
    "You are a coding expert. You will be given a coding problem, and you need to "
    "write a correct Python program that matches the specification and passes all tests. "
    "You may start by outlining your thought process. In the end, please provide the "
    "complete code in a code block enclosed with ``` ```.\n\n{problem}"
)


def parse_description(problem: str) -> str:
    """Extract problem description (before examples/input section)."""
    import re
    separators = re.compile(
        r"^[ \t]*(?:\#[ \t]*)*[ \t]*(?:input|example|-----)",
        re.IGNORECASE | re.MULTILINE,
    )
    m = separators.search(problem)
    return problem[:m.start()].strip() if m else problem.strip()


def decode_private_test_cases(encoded_data: str, fn_name: str) -> str:
    """Decode base64+zlib+pickle encoded test cases from LiveCodeBench."""
    decoded = base64.b64decode(encoded_data)
    decompressed = zlib.decompress(decoded)
    original = pickle.loads(decompressed)
    tests = json.loads(original)
    return json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "testtype": tests[0]["testtype"] if tests else "functional",
        "fn_name": fn_name,
        "time_limit": TIME_LIMIT,
    }, ensure_ascii=False)


def reduce_test_cases(tests_json: str, keep_ratio: float = 0.5, seed: int = 42) -> str:
    """Randomly keep a fraction of test cases (for training)."""
    tests = json.loads(tests_json)
    inputs = tests["inputs"]
    outputs = tests["outputs"]

    n = len(inputs)
    keep_count = max(1, int(n * keep_ratio))
    rng = np.random.RandomState(seed)
    keep_idx = np.sort(rng.choice(n, size=keep_count, replace=False))

    reduced = copy.deepcopy(tests)
    reduced["inputs"] = [inputs[i] for i in keep_idx]
    reduced["outputs"] = [outputs[i] for i in keep_idx]
    return json.dumps(reduced, ensure_ascii=False)


def process_livecodebench(seed: int = 42):
    """Load and process LiveCodeBench code_generation_lite."""
    print("Loading LiveCodeBench code_generation_lite from HuggingFace...")
    ds = datasets.load_dataset("livecodebench/code_generation_lite", split="test")
    print(f"  Loaded {len(ds)} problems")

    train_records = []
    test_records = []
    skipped = 0

    for example in ds:
        question_content = example.get("question_content", "")
        starter_code = example.get("starter_code", "") or ""
        private_test_cases = example.get("private_test_cases", "")
        contest_date = example.get("contest_date", None)
        metadata = example.get("metadata", "")

        if not question_content or not private_test_cases:
            skipped += 1
            continue

        # Parse function name from metadata
        fn_name = ""
        if metadata and metadata.strip():
            try:
                fn_name = json.loads(metadata).get("func_name", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # Build problem text
        problem = question_content
        if starter_code.strip():
            sig = "def " + starter_code.split("def ")[1].split("\n")[0] if "def " in starter_code else starter_code
            problem += f"\n\nYour solution should have the following signature: ```python\n{sig}\n```"

        # Remove 'self' from method signatures
        problem = problem.replace("(self, ", "(")

        # Decode test cases
        try:
            full_tests = decode_private_test_cases(private_test_cases, fn_name)
        except Exception:
            skipped += 1
            continue

        description = parse_description(problem)
        prompt_content = CODE_PROMPT.format(problem=problem)

        record = {
            "data_source": "livecodebench",
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "code",
            "reward_model": {"style": "code", "ground_truth": full_tests},
            "extra_info": {"description": description, "problem": problem},
            "pi_fields": {
                "problem": problem,
                "starter_code": starter_code,
                "test_cases": full_tests,
            },
        }

        # Time-based split
        is_train = True
        if contest_date is not None:
            if isinstance(contest_date, str):
                try:
                    contest_date = datetime.fromisoformat(contest_date)
                except (ValueError, TypeError):
                    pass
            if isinstance(contest_date, datetime) and contest_date >= TRAIN_CUTOFF:
                is_train = False

        if is_train:
            # Train: use reduced test cases (50%)
            train_record = copy.deepcopy(record)
            reduced_tests = reduce_test_cases(full_tests, keep_ratio=TEST_CASES_KEEP_RATIO, seed=seed)
            train_record["reward_model"]["ground_truth"] = reduced_tests
            train_record["extra_info"]["split"] = "train"
            train_records.append(train_record)
        else:
            # Test: use full test cases
            record["extra_info"]["split"] = "test"
            test_records.append(record)

    if skipped:
        print(f"  Skipped {skipped} samples")
    print(f"  Train: {len(train_records)} (before cutoff {TRAIN_CUTOFF.date()}, 50% tests)")
    print(f"  Test:  {len(test_records)} (after cutoff, full tests)")
    return train_records, test_records


def main():
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench v6 for OPSD")
    parser.add_argument("--output-dir", type=str, default="data/livecodebench")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, test_records = process_livecodebench(seed=args.seed)

    pd.DataFrame(train_records).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(test_records).to_parquet(output_dir / "test.parquet")

    print(f"  Saved: {output_dir / 'train.parquet'}, {output_dir / 'test.parquet'}")
    print("Done!")


if __name__ == "__main__":
    main()
