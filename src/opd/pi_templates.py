"""
PI (Privileged Information) template system for OPSD teacher prompts.

Templates are keyed by data_source. Each template is a format string with
{placeholders} that map to keys in the sample's pi_fields dict.

Usage:
    from opd.pi_templates import render_pi

    teacher_content = render_pi(
        data_source="openthoughts",
        pi_fields={"problem": "What is 2+2?", "solution": "2+2=4"},
    )
"""

from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Default PI Templates
# Placeholders map to pi_fields keys. Use {{}} to escape literal braces.
# ---------------------------------------------------------------------------

PI_TEMPLATES: dict[str, str] = {
    "openthoughts": (
        "{problem}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution ===\n{solution}\n=== End ===\n\n"
        "After reading the reference solution above, derive the same final answer "
        "using your own independent reasoning. Think step by step.\n"
        "Put your final answer within \\boxed{{}}."
    ),
    "math_dapo": (
        "{problem}\n\n"
        "The correct final answer to this problem is: {answer}\n\n"
        "Using this answer as guidance, derive the complete reasoning step by step.\n"
        "Put your final answer within \\boxed{{}}."
    ),
    "livecodebench": (
        "{problem}\n\n"
        "Here are the test cases for this problem:\n"
        "{test_cases}\n\n"
        "Write a correct and efficient Python solution that passes all test cases."
    ),
    "sciknoweval": (
        "{question}\n\n{choices}\n\n"
        "The correct answer is: {answer}\n\n"
        "Provide your reasoning and select the correct option."
    ),
    "tooluse": (
        "{prompt}\n\n"
        "The correct tool calls for this task are:\n{answer}\n\n"
        "Now solve the task using the correct actions."
    ),
}


def register_pi_template(data_source: str, template: str) -> None:
    """Register or override a PI template for a data_source."""
    PI_TEMPLATES[data_source] = template


def render_pi(
    data_source: str,
    pi_fields: dict[str, str],
    custom_templates: Optional[dict[str, str]] = None,
) -> str:
    """Render PI content for the teacher prompt.

    Looks up the template by data_source (custom_templates override defaults),
    then fills {placeholders} from pi_fields. Missing fields render as "".

    Args:
        data_source: Dataset identifier (e.g. "openthoughts", "math_dapo").
        pi_fields: Dict of field values for this sample.
        custom_templates: Optional override templates (e.g. from YAML config).

    Returns:
        Rendered teacher prompt content string.

    Raises:
        KeyError: If no template found for data_source.
    """
    templates = {**PI_TEMPLATES, **(custom_templates or {})}

    if data_source not in templates:
        raise KeyError(
            f"No PI template for data_source='{data_source}'. "
            f"Available: {list(templates.keys())}. "
            f"Register one with register_pi_template() or pass custom_templates."
        )

    template = templates[data_source]
    safe_fields = defaultdict(lambda: "", pi_fields or {})
    return template.format_map(safe_fields)
