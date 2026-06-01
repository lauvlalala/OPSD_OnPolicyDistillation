"""
Reward function for SciKnowEval: multiple-choice answer extraction.

Expects model to output answer in <answer>A/B/C/D</answer> format.
"""

import re


def extract_xml_answer(text: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text)
    if match:
        return match.group(1)
    # Fallback: last occurrence of <answer>...</answer>
    parts = text.split("<answer>")
    if len(parts) >= 2:
        answer_part = parts[-1].split("</answer>")[0].strip()
        if answer_part in ("A", "B", "C", "D"):
            return answer_part
    return None


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for SciKnowEval MCQ.

    Returns:
        dict with score, acc, pred, feedback fields.
    """
    if not solution_str:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Empty response."}

    predicted = extract_xml_answer(solution_str)

    if predicted is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "feedback": "Could not extract answer in <answer>A/B/C/D</answer> format.",
        }

    is_correct = predicted == ground_truth
    feedback = "" if is_correct else f"Incorrect. Expected {ground_truth}, got {predicted}."

    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": 1.0 if is_correct else 0.0,
        "pred": predicted,
        "feedback": feedback,
    }
