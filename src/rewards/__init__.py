"""
Unified reward function dispatch by data_source.

This module provides a single compute_score entry point that routes
to the appropriate dataset-specific reward function based on data_source.

Usage in verl config:
    reward.custom_reward_function.path = src/rewards/__init__.py
    reward.custom_reward_function.name = compute_score
"""

from typing import Union


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> Union[float, dict]:
    """Unified reward function that dispatches by data_source.

    Args:
        solution_str: Model's response text.
        ground_truth: Ground truth (format depends on data_source).
        **kwargs: May contain data_source, extra_info, etc.

    Returns:
        dict with at least {score, acc, pred, feedback} fields,
        or float for backward compatibility.
    """
    data_source = kwargs.get("data_source", "")
    if not data_source:
        extra_info = kwargs.get("extra_info", {})
        if isinstance(extra_info, dict):
            data_source = extra_info.get("data_source", "")

    if data_source in ("sciknoweval",):
        from rewards.sciknoweval import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("tooluse",):
        from rewards.tooluse import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("livecodebench", "code"):
        from rewards.livecodebench import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("searchqa",):
        from rewards.searchqa import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("alfworld",):
        from rewards.alfworld import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("webshop",):
        from rewards.webshop import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("swe_gym",):
        from rewards.swe import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    elif data_source in ("openthoughts", "math_dapo", "math", "math500"):
        from rewards.math_reward import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)

    else:
        from rewards.math_reward import compute_score as _fn
        return _fn(solution_str, ground_truth, **kwargs)
