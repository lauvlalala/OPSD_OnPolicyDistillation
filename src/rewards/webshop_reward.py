"""
Reward function for WebShop: task score from environment.

WebShop returns a reward between 0 and 1 based on how well the
purchased item matches the user's instruction.
"""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for WebShop.

    In multi-turn mode, the actual reward comes from WebShopTool.calc_reward().
    This is a fallback for offline evaluation.
    """
    try:
        score = float(ground_truth) if ground_truth else 0.0
    except (ValueError, TypeError):
        score = 0.0

    return {"score": score, "acc": 1.0 if score >= 1.0 else 0.0, "pred": "", "feedback": ""}
