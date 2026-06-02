"""
Reward function for SWE-Gym: test pass/fail.

In multi-turn mode, the actual reward comes from SWEBashTool.calc_reward()
which runs pytest inside the Docker container. This is a fallback for
offline evaluation or when tool reward is not available.
"""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for SWE-Gym.

    The real reward is determined by running tests in the Docker container
    (handled by SWEBashTool.calc_reward). This function is a fallback.
    """
    # In agent loop mode, reward comes from tool's calc_reward
    # This fallback just returns 0 since we can't run tests offline
    try:
        score = float(ground_truth) if ground_truth else 0.0
    except (ValueError, TypeError):
        score = 0.0

    return {"score": score, "acc": score, "pred": "", "feedback": ""}
