"""
Reward function for ALFWorld: task completion.

The reward comes from the environment (1.0 if task completed, 0.0 otherwise).
This function also extracts feedback from the trajectory for PI building.
"""

import re


def _extract_feedback(trajectory: str) -> str:
    """Extract failure feedback from ALFWorld trajectory.

    Looks for repeated "Nothing happens" or the last few observations
    to summarize what went wrong.
    """
    if not trajectory:
        return ""

    # Extract tool/observation responses (between assistant actions)
    observations = re.findall(r"(?:Observation|observation|Output):\s*(.+?)(?:\n|$)", trajectory)
    if not observations:
        # Fallback: get lines that look like environment responses
        lines = trajectory.split("\n")
        observations = [l.strip() for l in lines if l.strip() and not l.strip().startswith(("{", "\"command", "\"action"))]

    if not observations:
        return ""

    # Count "Nothing happens" — a strong signal of wrong actions
    nothing_count = sum(1 for o in observations if "Nothing happens" in o)

    # Get last 3 observations as context
    last_obs = observations[-3:] if len(observations) >= 3 else observations
    summary_parts = []

    if nothing_count > 0:
        summary_parts.append(f"Agent took {nothing_count} invalid action(s) ('Nothing happens').")

    summary_parts.append("Last observations: " + " | ".join(o[:100] for o in last_obs))

    return " ".join(summary_parts)


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for ALFWorld.

    In multi-turn mode, the actual reward comes from AlfWorldTool.calc_reward().
    This function extracts feedback from the trajectory for PI.
    """
    try:
        score = float(ground_truth) if ground_truth else 0.0
    except (ValueError, TypeError):
        score = 0.0

    feedback = ""
    if score < 1.0 and solution_str:
        feedback = _extract_feedback(solution_str)

    return {"score": score, "acc": score, "pred": "", "feedback": feedback}
