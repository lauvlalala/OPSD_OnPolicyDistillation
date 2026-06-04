"""
OPD Worker: extends verl's ActorRolloutRefWorker with divergence-based training.

A separate teacher model (ref) and trainable student model (actor) see the same
input sequences. The teacher produces better distributions naturally (bigger/stronger).

Training step:
  1. Forward teacher (ref model, frozen) on teacher_input_ids -> teacher logits
  2. Forward student (actor model, trainable) on student_input_ids -> student logits
  3. Compute token-wise divergence along response positions
  4. Backward and step optimizer
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import entropy_from_logits_with_chunking
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .losses import LOSS_FN_MAP, compute_sampled_token_loss, compute_teacher_token_stats, compute_topk_divergence_loss

logger = logging.getLogger(__name__)


def _shift_loss_mask_right_per_sequence(loss_mask_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Shift loss-mask labels within each unpadded sequence only.

    The padded path uses ``loss_mask[:, 1:]`` so each logit at position ``t``
    is trained against whether token ``t + 1`` belongs to the response. After
    unpadding, we must preserve that same next-token alignment without letting
    the final token of one sample spill into the first token of the next.
    """
    shifted = torch.zeros_like(loss_mask_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start = int(seq_start.item())
        end = int(seq_end.item())
        if end - start > 1:
            shifted[start : end - 1] = loss_mask_rmpad[start + 1 : end]
    return shifted


def _shift_ids_within_sequences(ids_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Get next-token target IDs within each unpadded sequence.

    For each sequence, target[t] = ids[t+1]. The last token of each sequence
    gets 0 (won't be selected by loss_mask anyway).
    """
    shifted = torch.zeros_like(ids_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        s, e = int(seq_start.item()), int(seq_end.item())
        if e - s > 1:
            shifted[s : e - 1] = ids_rmpad[s + 1 : e]
    return shifted


class OPDWorker(AsyncActorRolloutRefWorker):
    """Worker with on-policy distillation training on top of verl's actor+rollout+ref.

    Inherits actor_module_fsdp (student) and ref_module_fsdp (teacher) from
    the HybridEngine ActorRolloutRef worker. Both teacher and student see the
    same input sequences (no privileged information).
    """

    def __init__(self, config, role: str, **kwargs):
        # The parent trainer's init_workers() creates hybrid-engine workers with
        # role="actor_rollout", which sets _is_ref=False and skips building
        # ref_module_fsdp.  OPD needs the colocated ref model as the teacher,
        # so we promote the role to "actor_rollout_ref".
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def sync_ref_from_actor(self, data: DataProto) -> DataProto:
        """Sync teacher (ref) weights from student (actor). Supports hard copy and EMA.

        Handles verl 0.7.0's forced native CPUOffload on ref by operating on
        whatever device each param is on (CPU for native-offloaded ref, GPU for actor).

        Args (via data.meta_info):
            ema_decay: 0 = hard copy, >0 = EMA: ref = decay*ref + (1-decay)*actor
        """
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        ema_decay = data.meta_info.get("ema_decay", 0.0)

        # Actor uses manual offload; bring to GPU so we can read its params
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        with torch.no_grad():
            for (ref_name, ref_p), (_, actor_p) in zip(
                self.ref_module_fsdp.named_parameters(),
                self.actor_module_fsdp.named_parameters(),
            ):
                # Move actor param to ref's device (handles native CPUOffload on ref)
                actor_data = actor_p.data.to(ref_p.data.device)
                if ema_decay > 0:
                    ref_p.data.mul_(ema_decay).add_(actor_data, alpha=1 - ema_decay)
                else:
                    ref_p.data.copy_(actor_data)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        logger.info("[OPD] sync_ref_from_actor: ema_decay=%.4f", ema_decay)
        return DataProto()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_opd(self, data: DataProto) -> DataProto:
        """One OPD training step: divergence between teacher and student.

        Uses a two-phase approach to avoid holding both models on GPU simultaneously:
          Phase 1: Load teacher (ref) → run all teacher forwards → cache logits on CPU → offload teacher
          Phase 2: Load student (actor) + optimizer → train with cached teacher logits → offload

        Input DataProto batch keys:
          teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        Meta info:
          opd_loss_type: "reverse_kl" | "forward_kl" | "jsd"
          opd_beta: JSD interpolation (only used for jsd)
          opd_chunk_size: tokens per chunk for loss computation
        """
        assert self._is_actor, "update_opd requires actor role"

        def _mem(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        ref_needs_offload = self.config.ref.fsdp_config.get("param_offload", False)

        data = data.to("cpu")
        loss_type = data.meta_info.get("opd_loss_type", "reverse_kl")
        beta = data.meta_info.get("opd_beta", 0.5)
        chunk_size = data.meta_info.get("opd_chunk_size", 512)
        token_scope = data.meta_info.get("opd_token_scope", "full_vocab")
        temperature = data.meta_info.get("opd_temperature", 1.0)
        token_clip = data.meta_info.get("opd_token_clip", 0.0)
        topk_k = data.meta_info.get("opd_topk", 64)
        weight_mode = data.meta_info.get("opd_weight_mode", "reinforce")
        gate_beta = data.meta_info.get("opd_gate_beta", 5.0)

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )
        device = get_device_id()

        batch_size = data.batch["student_input_ids"].shape[0]
        single_model_mode = data.meta_info.get("single_model_mode", False)
        if batch_size == 0:
            return DataProto(meta_info={"metrics": {"opd/loss": 0.0, "opd/num_tokens": 0}})

        micro_batches = list(data.split(micro_batch_size))
        logger.info("[OPD-MEM] batch_size=%d, micro_batches=%d, micro_batch_size=%d, ref_needs_offload=%s",
                     batch_size, len(micro_batches), micro_batch_size, ref_needs_offload)
        _mem("before-ulysses")

        with self.ulysses_sharding_manager:
            # ------------------------------------------------------------------
            # Phase 1: Teacher forward
            # ------------------------------------------------------------------
            _mem("phase1-before-ref-load")
            if single_model_mode:
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                teacher_model = self.actor_module_fsdp
            else:
                if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None and ref_needs_offload:
                    load_fsdp_model_to_gpu(self.ref_module_fsdp)
                teacher_model = self.ref_module_fsdp
            _mem("phase1-after-ref-load")

            teacher_model.eval()
            forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded
            teacher_logits_cache = []  # list of (teacher_logits_cpu, is_valid)

            for i, micro_batch in enumerate(micro_batches):
                micro_batch = micro_batch.to(device)
                t_input_ids = micro_batch.batch["teacher_input_ids"]
                t_attention_mask = micro_batch.batch["teacher_attention_mask"]
                t_position_ids = micro_batch.batch["teacher_position_ids"]
                t_loss_mask = micro_batch.batch["teacher_loss_mask"]

                logger.info("[OPD-MEM] phase1 micro_batch[%d]: teacher_ids shape=%s, loss_mask response_tokens=%d",
                            i, list(t_input_ids.shape), int(t_loss_mask[:, 1:].sum().item()))
                _mem(f"phase1-mb{i}-before-teacher-fwd")

                with torch.no_grad():
                    teacher_logits = forward_fn(
                        teacher_model, t_input_ids, t_attention_mask, t_position_ids, t_loss_mask
                    )
                logger.info("[OPD-MEM] phase1 micro_batch[%d]: teacher_logits shape=%s",
                            i, list(teacher_logits.shape))
                _mem(f"phase1-mb{i}-after-teacher-fwd")

                # Cache based on token_scope for memory efficiency
                if token_scope == "sampled_token":
                    # Only cache (N,) teacher log-probs at actual next-token positions
                    if use_remove_padding:
                        target_ids = self._extract_response_target_ids_unpadded(
                            t_input_ids, t_attention_mask, t_loss_mask
                        )
                    else:
                        target_ids = self._extract_response_target_ids_padded(t_input_ids, t_loss_mask)
                    t_lp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
                    teacher_lp_at_target = t_lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                    teacher_logits_cache.append(((teacher_lp_at_target.to("cpu"), target_ids.to("cpu")), True))
                    del t_lp, teacher_lp_at_target, target_ids
                elif token_scope == "topk":
                    # Only cache (N, k) teacher top-k logits + indices
                    topk_vals, topk_idx = teacher_logits.topk(topk_k, dim=-1)
                    teacher_logits_cache.append(((topk_vals.to("cpu"), topk_idx.to("cpu")), True))
                    del topk_vals, topk_idx
                else:
                    # full_vocab: cache entire (N, V) logits
                    teacher_logits_cache.append((teacher_logits.to("cpu"), True))
                del teacher_logits
                _mem(f"phase1-mb{i}-after-cache-to-cpu")

            _mem("phase1-before-ref-offload")
            if not single_model_mode and ref_needs_offload:
                offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            _mem("phase1-after-ref-offload")

            if not single_model_mode:
                torch.cuda.empty_cache()
            _mem("phase1-after-empty-cache")

            # ------------------------------------------------------------------
            # Phase 2: Student forward + loss + backward — actor + optimizer on GPU
            # ------------------------------------------------------------------
            if not single_model_mode and self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            _mem("phase2-after-actor-load")
            if self._is_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=device)
            _mem("phase2-after-optimizer-load")

            use_sample_weights = "sample_weights" in data.batch
            metrics = self._opd_training_step(
                micro_batches, teacher_logits_cache,
                loss_type=loss_type, beta=beta, chunk_size=chunk_size,
                use_remove_padding=use_remove_padding, device=device, batch_size=batch_size,
                use_sample_weights=use_sample_weights,
                token_scope=token_scope, temperature=temperature, token_clip=token_clip,
                topk_k=topk_k, weight_mode=weight_mode, gate_beta=gate_beta,
            )

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["opd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            if metrics["opd/num_tokens"] > 0:
                self.actor_lr_scheduler.step()
            else:
                metrics["opd/skipped_step"] = 1.0

            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    def _opd_training_step(
        self,
        micro_batches: list,
        teacher_logits_cache: list,
        loss_type: str = "reverse_kl",
        beta: float = 0.5,
        chunk_size: int = 512,
        use_remove_padding: bool = False,
        device: int = 0,
        batch_size: int = 0,
        use_sample_weights: bool = False,
        token_scope: str = "full_vocab",
        temperature: float = 1.0,
        token_clip: float = 0.0,
        topk_k: int = 64,
        weight_mode: str = "reinforce",
        gate_beta: float = 5.0,
    ) -> dict:
        """Core OPD training with pre-computed teacher logits.

        Teacher logits are already cached on CPU from phase 1. This phase only
        has the student (actor) model + optimizer on GPU, avoiding OOM from
        holding both models simultaneously.

        For each micro-batch:
          1. Move cached teacher logits to GPU
          2. Student forward (trainable actor) on student_input_ids -> logits
          3. Compute divergence loss (chunk-wise for memory efficiency)
          4. Backward with gradient accumulation scaling
        """
        self.actor_module_fsdp.train()

        active_micro_batches = sum(1 for _, is_valid in teacher_logits_cache if is_valid)
        grad_accum = max(1, active_micro_batches)

        loss_fn = LOSS_FN_MAP[loss_type]
        forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_entropy_sum = 0.0
        total_tokens = 0
        total_rows = 0

        def _mem2(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        for i, (micro_batch, (cached_data, is_valid)) in enumerate(
            zip(micro_batches, teacher_logits_cache, strict=True)
        ):
            if not is_valid:
                continue

            micro_batch = micro_batch.to(device)

            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            _mem2(f"phase2-mb{i}-before-teacher-to-gpu")

            # Student forward (always needed)
            student_logits = forward_fn(
                self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask
            )
            _mem2(f"phase2-mb{i}-after-student-fwd")

            n_response_tokens = student_logits.shape[0]
            if n_response_tokens == 0:
                continue

            _mem2(f"phase2-mb{i}-before-loss")

            # --- Compute loss based on token_scope ---
            if token_scope == "sampled_token":
                teacher_lp_cpu, target_ids_cpu = cached_data
                teacher_lp = teacher_lp_cpu.to(device)
                target_ids = target_ids_cpu.to(device)

                if teacher_lp.shape[0] != n_response_tokens:
                    raise RuntimeError("Teacher/student response token count mismatch (sampled_token)")

                # Gather student log-prob at target positions (with grad)
                student_lp = F.log_softmax(student_logits.float() / temperature, dim=-1)
                student_lp_at_target = student_lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                del student_lp

                loss, n_tokens, extra_metrics = compute_sampled_token_loss(
                    student_lp_at_target, teacher_lp,
                    weight_mode=weight_mode, gate_beta=gate_beta,
                )
                del teacher_lp, target_ids, student_lp_at_target

            elif token_scope == "topk":
                topk_vals_cpu, topk_idx_cpu = cached_data
                teacher_topk_logits = topk_vals_cpu.to(device)  # (N, k)
                topk_idx = topk_idx_cpu.to(device)  # (N, k)

                if teacher_topk_logits.shape[0] != n_response_tokens:
                    raise RuntimeError("Teacher/student response token count mismatch (topk)")

                n_tokens = n_response_tokens
                # Gather student logits at teacher's top-k positions
                student_topk_logits = student_logits.gather(-1, topk_idx)  # (N, k)
                # Renormalize both over k tokens and compute divergence
                t_lp = F.log_softmax(teacher_topk_logits.float() / temperature, dim=-1)
                s_lp = F.log_softmax(student_topk_logits.float() / temperature, dim=-1)
                del teacher_topk_logits, student_topk_logits, topk_idx

                if loss_type == "forward_kl":
                    div_per_vocab = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
                elif loss_type == "reverse_kl":
                    div_per_vocab = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
                else:
                    # JSD on top-k
                    log_m = torch.logsumexp(
                        torch.stack([t_lp + math.log(beta), s_lp + math.log(1 - beta)], dim=0), dim=0
                    )
                    kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True)
                    kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True)
                    div_per_vocab = beta * kl_t + (1 - beta) * kl_s
                    del log_m, kl_t, kl_s
                del t_lp, s_lp

                if token_clip > 0:
                    div_per_vocab = div_per_vocab.clamp(max=token_clip)
                loss = div_per_vocab.sum(dim=-1).mean()
                del div_per_vocab
                extra_metrics = {}

            else:
                # full_vocab path (original behavior)
                teacher_logits = cached_data.to(device)

                if teacher_logits.shape[0] != n_response_tokens:
                    raise RuntimeError("Teacher/student response token count mismatch (full_vocab)")

                # Per-sample reward weighting
                mb_sample_weights = None
                if use_sample_weights and "sample_weights" in micro_batch.batch:
                    mb_sample_weights = micro_batch.batch["sample_weights"].to(device)

                # Resolve loss function and kwargs
                loss_kwargs = {"chunk_size": chunk_size, "temperature": temperature, "token_clip": token_clip}
                if loss_type == "jsd":
                    loss_kwargs["beta"] = beta
                _loss_fn = loss_fn

                if mb_sample_weights is not None:
                    per_sample_token_counts = s_loss_mask[:, 1:].sum(dim=1).long()
                    sample_losses = []
                    token_offset = 0
                    for si in range(per_sample_token_counts.shape[0]):
                        n_tok = per_sample_token_counts[si].item()
                        if n_tok == 0:
                            sample_losses.append(torch.zeros(1, device=device, requires_grad=True).squeeze())
                            token_offset += n_tok
                            continue
                        t_slice = teacher_logits[token_offset:token_offset + n_tok]
                        s_slice = student_logits[token_offset:token_offset + n_tok]
                        sl, _ = _loss_fn(t_slice, s_slice, **loss_kwargs)
                        sample_losses.append(sl)
                        token_offset += n_tok
                    sample_losses = torch.stack(sample_losses)
                    weight_sum = mb_sample_weights.sum()
                    loss = (sample_losses * mb_sample_weights).sum() / weight_sum.clamp(min=1e-8)
                    n_tokens = int(per_sample_token_counts.sum().item())
                else:
                    loss, n_tokens = _loss_fn(teacher_logits, student_logits, **loss_kwargs)

                del teacher_logits
                extra_metrics = {}

            _mem2(f"phase2-mb{i}-after-loss")

            # Compute per-token entropy of the student policy (no grad needed)
            with torch.no_grad():
                token_entropy = entropy_from_logits_with_chunking(student_logits.float(), chunk_size=chunk_size)
                total_entropy_sum += token_entropy.sum().item()

            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            _mem2(f"phase2-mb{i}-after-backward")

            total_loss += loss.detach().item()
            total_tokens += n_tokens
            total_rows += s_input_ids.shape[0]

        if total_tokens == 0:
            self.actor_optimizer.zero_grad()
            return {
                "opd/loss": 0.0,
                "opd/entropy": 0.0,
                "opd/grad_norm": 0.0,
                "opd/num_tokens": 0,
                "opd/batch_size": int(batch_size),
                "opd/valid_rows": int(total_rows),
            }

        # Gradient clipping and optimizer step
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module_fsdp.parameters(), max_norm=grad_clip
            )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        return {
            "opd/loss": total_loss / max(1, active_micro_batches),
            "opd/entropy": total_entropy_sum / max(1, total_tokens),
            "opd/grad_norm": grad_norm.detach().item(),
            "opd/num_tokens": int(total_tokens),
            "opd/batch_size": batch_size,
            "opd/valid_rows": int(total_rows),
        }

    def _extract_response_target_ids_padded(self, input_ids, loss_mask):
        """Extract next-token target IDs at response positions (padded path)."""
        target_ids = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:]
        flat_target = target_ids.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        return flat_target[flat_mask.nonzero(as_tuple=True)[0]]

    def _extract_response_target_ids_unpadded(self, input_ids, attention_mask, loss_mask):
        """Extract next-token target IDs at response positions (unpadded path)."""
        _, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        ids_rmpad = input_ids.reshape(-1)[indices]
        target_ids_rmpad = _shift_ids_within_sequences(ids_rmpad, cu_seqlens)
        loss_mask_rmpad = loss_mask.reshape(-1)[indices]
        shifted_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        return target_ids_rmpad[shifted_mask.nonzero(as_tuple=True)[0]]

    def _forward_logits_padded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (padded path).

        Returns: (N_response_tokens, vocab_size)
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits
            del outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_loss_mask = loss_mask[:, 1:]
        del logits

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_logits[response_indices]

    def _forward_logits_unpadded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (unpadded/flash_attn path).

        Returns: (N_response_tokens, vocab_size)
        """
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad,
                position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )

        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]

        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        return logits_rmpad[response_indices]
