"""
Reward function for ALFWorld: task completion.

The reward comes from the environment (1.0 if task completed, 0.0 otherwise).
This function is a pass-through for the env reward stored in ground_truth.
"""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for ALFWorld.

    In multi-turn mode, the actual reward comes from AlfWorldTool.calc_reward().
    This function handles the case where reward is passed via ground_truth.
    """
    # In agent loop mode, reward is computed by the tool, not this function.
    # This is a fallback for offline evaluation.
    try:
        score = float(ground_truth) if ground_truth else 0.0
    except (ValueError, TypeError):
        score = 0.0

    return {"score": score, "acc": score, "pred": "", "feedback": ""}
