"""
Reward function for ToolUse: Action/Input sequence matching.

Adapted from SDPO's verl/utils/reward_score/feedback/tooluse.py.
Evaluates both action names and action inputs for correctness.
"""

import json
import re
from collections import Counter


def extract_actions(text: str) -> list[str]:
    """Extract all action names after 'Action:' occurrences."""
    return re.findall(r"Action:\s*(\w+)", text)


def extract_action_inputs(text: str) -> dict:
    """Extract and merge all JSON blocks following 'Action Input:'."""
    json_blocks = re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL)
    combined = {}
    for block in json_blocks:
        try:
            combined.update(json.loads(block))
        except json.JSONDecodeError:
            pass
    return combined


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for ToolUse task.

    Args:
        solution_str: The model's response text.
        ground_truth: JSON string of expected actions, e.g.:
            '[{"Action": "search", "Action_Input": "{\"query\": \"test\"}"}]'

    Returns:
        dict with score, acc, pred, feedback fields.
    """
    if not solution_str or not ground_truth:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Empty input."}

    # Parse ground truth
    try:
        gt_list = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError):
        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Failed to parse ground truth."}

    if not isinstance(gt_list, list):
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Ground truth is not a list."}

    # Extract ground truth actions and inputs
    gt_actions = [item["Action"] for item in gt_list if "Action" in item]
    gt_inputs = {}
    for item in gt_list:
        try:
            inp = item.get("Action_Input", "{}")
            parsed = json.loads(inp) if isinstance(inp, str) else inp
            if isinstance(parsed, dict):
                gt_inputs.update(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    # Extract predicted actions and inputs
    pred_actions = extract_actions(solution_str)
    pred_inputs = extract_action_inputs(solution_str)

    # Check correctness
    actions_correct = Counter(pred_actions) == Counter(gt_actions)
    inputs_correct = pred_inputs == gt_inputs
    is_correct = actions_correct and inputs_correct

    # Build feedback
    feedback_parts = []
    if not pred_actions:
        feedback_parts.append("No Action: found in response.")
    elif not actions_correct:
        feedback_parts.append(f"Actions mismatch: predicted {pred_actions}, expected {gt_actions}.")
    if not inputs_correct:
        feedback_parts.append(f"Inputs mismatch: predicted keys {list(pred_inputs.keys())}, expected {list(gt_inputs.keys())}.")

    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": 1.0 if is_correct else 0.0,
        "pred": f"Actions: {pred_actions}",
        "feedback": " ".join(feedback_parts) if feedback_parts else "",
    }
