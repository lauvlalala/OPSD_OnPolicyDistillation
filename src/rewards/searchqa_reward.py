"""
Reward function for SearchQA: Exact Match on <answer> tag.

Adapted from Search-R1 style evaluation.
"""

import re
import string


def normalize_answer(s: str) -> str:
    """Lower, remove articles/punctuation/whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split()).strip()


def extract_answer(text: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def compute_score(solution_str: str, ground_truth, **kwargs) -> dict:
    """Compute EM reward for SearchQA.

    Args:
        ground_truth: dict with "target" key containing list of acceptable answers.
    """
    if not solution_str:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Empty response."}

    # Parse ground_truth
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", [])
    elif isinstance(ground_truth, list):
        targets = ground_truth
    else:
        targets = [str(ground_truth)]

    answer = extract_answer(solution_str)
    if answer is None:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "No <answer> tag found."}

    normalized_pred = normalize_answer(answer)
    for target in targets:
        if normalize_answer(str(target)) == normalized_pred:
            return {"score": 1.0, "acc": 1.0, "pred": answer, "feedback": ""}

    return {"score": 0.0, "acc": 0.0, "pred": answer, "feedback": f"Expected one of {targets[:3]}"}
