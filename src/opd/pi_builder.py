"""
PI (Privileged Information) builder for OPSD teacher prompts.

Supports three PI sources:
  1. Static: from pi_fields in data (ground truth, solution text, etc.)
  2. Rollout: from successful rollouts in the same batch (SDPO-style)
  3. Feedback: from reward function's feedback field (execution errors, etc.)

The builder constructs teacher messages that contain privileged context,
enabling the teacher to produce a better distribution for distillation.
"""

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


DEFAULT_REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
DEFAULT_SOLUTION_TEMPLATE = "\n\nCorrect solution:\n{successful_previous_attempt}\n"
DEFAULT_FEEDBACK_TEMPLATE = "\n\nFeedback from your earlier attempt:\n{feedback_raw}\n"


def _remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> tags and their content."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def collect_successful_rollouts(
    batch: DataProto,
    rewards: list[float],
    success_threshold: float = 0.5,
) -> dict:
    """Find successful rollouts grouped by uid.

    Args:
        batch: Rollout batch with non_tensor_batch["uid"].
        rewards: Per-sample reward scores.
        success_threshold: Minimum reward to count as success.

    Returns:
        Dict mapping uid → list of (sample_index, response_text_index) for successes.
    """
    uids = batch.non_tensor_batch.get("uid", np.array([]))
    success_by_uid = defaultdict(list)
    for idx, uid in enumerate(uids):
        if rewards[idx] >= success_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def build_pi_teacher_content(
    idx: int,
    batch: DataProto,
    responses_text: list[str],
    rewards: list[float],
    success_by_uid: dict,
    feedbacks: list[Optional[str]],
    pi_config: dict,
) -> Optional[str]:
    """Build teacher prompt content for a single sample using PI sources.

    Priority:
      1. If rollout PI enabled and a successful rollout exists → use it
      2. If feedback PI enabled and feedback exists (and no solution or only_without_solution=False) → use it
      3. If neither available → return None (skip this sample for distillation)

    Args:
        idx: Sample index in batch.
        batch: Full rollout batch.
        responses_text: Decoded response texts for all samples.
        rewards: Per-sample rewards.
        success_by_uid: Mapping from uid to list of successful indices.
        feedbacks: Per-sample feedback strings (from reward function).
        pi_config: PI configuration dict from opd config.

    Returns:
        Teacher prompt content string, or None if no PI available.
    """
    rollout_cfg = pi_config.get("pi_rollout", {})
    feedback_cfg = pi_config.get("pi_feedback", {})
    reprompt_template = pi_config.get("reprompt_template", DEFAULT_REPROMPT_TEMPLATE)
    solution_template = pi_config.get("solution_template", DEFAULT_SOLUTION_TEMPLATE)
    feedback_template = pi_config.get("feedback_template", DEFAULT_FEEDBACK_TEMPLATE)

    # Get original prompt text
    raw_prompts = batch.non_tensor_batch.get("raw_prompt", [])
    if idx < len(raw_prompts):
        # Extract last user message content as prompt
        msgs = raw_prompts[idx]
        prompt_text = ""
        for msg in reversed(msgs if isinstance(msgs, list) else []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = msg["content"]
                break
    else:
        prompt_text = ""

    # --- Source 1: Rollout PI ---
    solution_str = None
    if rollout_cfg.get("enable", False):
        uids = batch.non_tensor_batch.get("uid", np.array([]))
        uid = uids[idx] if idx < len(uids) else None
        if uid is not None:
            candidates = success_by_uid.get(uid, [])
            # Optionally exclude self
            if rollout_cfg.get("exclude_self", True):
                candidates = [j for j in candidates if j != idx]
            if candidates:
                solution_idx = candidates[0]
                solution_str = responses_text[solution_idx]
                if rollout_cfg.get("remove_thinking", True):
                    solution_str = _remove_thinking_trace(solution_str)

    # --- Source 2: Feedback PI ---
    feedback_str = None
    if feedback_cfg.get("enable", False):
        only_without_solution = feedback_cfg.get("only_without_solution", True)
        has_feedback = feedbacks[idx] is not None and feedbacks[idx].strip()
        if has_feedback and (not only_without_solution or solution_str is None):
            feedback_str = feedbacks[idx]

    # --- Build teacher content ---
    if solution_str is None and feedback_str is None:
        return None  # No PI available, skip

    solution_section = ""
    if solution_str is not None:
        solution_section = solution_template.format(successful_previous_attempt=solution_str)

    feedback_section = ""
    if feedback_str is not None:
        feedback_section = feedback_template.format(feedback_raw=feedback_str)

    teacher_content = reprompt_template.format(
        prompt=prompt_text,
        solution=solution_section,
        feedback=feedback_section,
    )
    return teacher_content


def build_pi_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    responses_text: list[str],
    rewards: list[float],
    feedbacks: list[Optional[str]],
    pi_config: dict,
    max_length: int = 16384,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> Optional[DataProto]:
    """Build teacher/student batch using rollout PI + feedback PI.

    This replaces the static PI path when pi_mode="rollout" or "rollout+feedback".

    Args:
        batch: Full rollout batch (with responses, response_mask, raw_prompt, uid).
        tokenizer: Tokenizer for encoding teacher prompts.
        responses_text: Decoded response texts.
        rewards: Per-sample reward scores.
        feedbacks: Per-sample feedback strings from reward function.
        pi_config: PI configuration from opd config.
        max_length: Max sequence length for teacher/student batch.
        apply_chat_template_kwargs: Chat template kwargs.

    Returns:
        DataProto with teacher_*/student_* tensors, or None if no samples have PI.
    """
    from common.batch_builder import _build_sequence_from_token_ids, _get_response_mask

    rollout_cfg = pi_config.get("pi_rollout", {})
    success_threshold = rollout_cfg.get("success_threshold", 0.5)

    # Collect successful rollouts
    success_by_uid = collect_successful_rollouts(batch, rewards, success_threshold)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    chat_kwargs = dict(apply_chat_template_kwargs or {})
    response_mask = _get_response_mask(batch)
    responses = batch.batch["responses"]

    student_seqs = []
    teacher_seqs = []
    kept_indices = []
    has_pi_mask = []
    skipped = 0

    raw_prompts = batch.non_tensor_batch.get("raw_prompt", [None] * len(batch))

    for i in range(len(batch)):
        # Get valid response IDs
        sample_response_mask = response_mask[i]
        valid_response_ids = responses[i][sample_response_mask.bool()]
        if valid_response_ids.numel() == 0:
            skipped += 1
            continue

        # Tokenize student prompt (original) — needed for both PI and non-PI cases
        student_messages = raw_prompts[i] if (i < len(raw_prompts) and raw_prompts[i] is not None) else []
        if not (isinstance(student_messages, list) and student_messages):
            skipped += 1
            continue

        student_prompt_ids = tokenizer.apply_chat_template(
            student_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
        )
        s_seq = _build_sequence_from_token_ids(student_prompt_ids, valid_response_ids, max_length, pad_token_id)
        if s_seq is None:
            skipped += 1
            continue

        # Build teacher content via PI
        teacher_content = build_pi_teacher_content(
            idx=i, batch=batch, responses_text=responses_text,
            rewards=rewards, success_by_uid=success_by_uid,
            feedbacks=feedbacks, pi_config=pi_config,
        )

        has_pi = teacher_content is not None
        if has_pi:
            # Tokenize teacher prompt with PI
            teacher_messages = []
            if i < len(raw_prompts) and raw_prompts[i] is not None:
                orig_msgs = raw_prompts[i] if isinstance(raw_prompts[i], list) else []
                for msg in orig_msgs:
                    if isinstance(msg, dict) and msg.get("role") == "system":
                        teacher_messages.append(msg)
            teacher_messages.append({"role": "user", "content": teacher_content})

            teacher_prompt_ids = tokenizer.apply_chat_template(
                teacher_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
            )
            t_seq = _build_sequence_from_token_ids(teacher_prompt_ids, valid_response_ids, max_length, pad_token_id)
            if t_seq is None:
                # PI too long, fall through to no-PI path
                has_pi = False

        if not has_pi:
            # No PI: use student prompt as teacher prompt, keep loss_mask matching
            t_seq = {k: v.clone() for k, v in s_seq.items()}

        teacher_seqs.append(t_seq)
        student_seqs.append(s_seq)
        kept_indices.append(i)
        has_pi_mask.append(has_pi)

    if skipped:
        logger.info("PI builder: skipped %d samples (empty response/prompt), kept %d", skipped, len(kept_indices))

    if not teacher_seqs:
        return None

    # Stack into DataProto
    batch_dict = {}
    for prefix, seqs in [("teacher_", teacher_seqs), ("student_", student_seqs)]:
        for key in ("input_ids", "attention_mask", "position_ids", "loss_mask"):
            batch_dict[f"{prefix}{key}"] = torch.stack([s[key] for s in seqs])
    batch_dict["valid_row_mask"] = torch.ones(len(student_seqs), dtype=torch.bool)
    # distillation_mask: per-sample weight (1.0 = has PI, 0.0 = no PI → loss zeroed)
    batch_dict["sample_weights"] = torch.tensor(
        [1.0 if m else 0.0 for m in has_pi_mask], dtype=torch.float32
    )

    return DataProto.from_single_dict(batch_dict)
