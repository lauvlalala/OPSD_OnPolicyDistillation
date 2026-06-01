"""
Chunk-wise memory-efficient divergence losses for OPD.

All loss functions process tokens in chunks to avoid OOM from materializing
full (N, V) float32 probability tensors. With V=152K (Qwen3), a single
(N, V) float32 tensor at N=4096 would be ~2.3 GB.

Supports three divergence types:
  - reverse_kl: KL(p_student || p_teacher) — mode-seeking
  - forward_kl: KL(p_teacher || p_student) — mean-seeking
  - jsd: JSD_beta(p_teacher || p_student) — interpolation of both
"""

import math

import torch
import torch.nn.functional as F


def compute_reverse_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int = 512,
    temperature: float = 1.0,
    token_clip: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise reverse KL: KL(p_student || p_teacher).

    Mode-seeking: student concentrates mass on teacher's high-probability
    tokens, adopting the teacher's concise reasoning style.

    KL(p_S || p_T) = sum_x p_S(x) * [log p_S(x) - log p_T(x)]

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        chunk_size: Tokens per chunk to bound peak memory.
        temperature: Divide logits by this before softmax (1.0 = no scaling).
        token_clip: Clamp per-element divergence to this value (0 = no clip).

    Returns:
        (loss, n_tokens) — scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    # Clone teacher chunks and free original for lower peak memory
    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    kl_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float() / temperature, dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float() / temperature, dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        # F.kl_div(input=log_p_T, target=log_p_S, log_target=True)
        # computes: exp(target) * (target - input) = p_S * (log_p_S - log_p_T)
        kl_per_vocab = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
        del t_lp, s_lp
        if token_clip > 0:
            kl_per_vocab = kl_per_vocab.clamp(max=token_clip)
        kl_chunk = kl_per_vocab.sum(dim=-1)
        del kl_per_vocab

        kl_sum = kl_sum + kl_chunk.sum()
        del kl_chunk

    return kl_sum / n_tokens, n_tokens


def compute_forward_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int = 512,
    temperature: float = 1.0,
    token_clip: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise forward KL: KL(p_teacher || p_student).

    Mean-seeking: student spreads probability to cover all modes of the
    teacher distribution, avoiding zero probability on teacher-likely tokens.

    KL(p_T || p_S) = sum_x p_T(x) * [log p_T(x) - log p_S(x)]

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        chunk_size: Tokens per chunk.
        temperature: Divide logits by this before softmax (1.0 = no scaling).
        token_clip: Clamp per-element divergence to this value (0 = no clip).

    Returns:
        (loss, n_tokens) — scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    kl_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float() / temperature, dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float() / temperature, dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        # F.kl_div(input=log_p_S, target=log_p_T, log_target=True)
        # computes: exp(target) * (target - input) = p_T * (log_p_T - log_p_S)
        kl_per_vocab = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
        del t_lp, s_lp
        if token_clip > 0:
            kl_per_vocab = kl_per_vocab.clamp(max=token_clip)
        kl_chunk = kl_per_vocab.sum(dim=-1)
        del kl_per_vocab

        kl_sum = kl_sum + kl_chunk.sum()
        del kl_chunk

    return kl_sum / n_tokens, n_tokens


def compute_jsd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.5,
    chunk_size: int = 256,
    temperature: float = 1.0,
    token_clip: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise Jensen-Shannon divergence with logsumexp mixture.

    JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
    where m = beta * p_T + (1-beta) * p_S

    The mixture is computed in log-space via logsumexp to avoid materializing
    explicit probability tensors.

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        beta: Interpolation weight (0.5 = symmetric JSD).
        chunk_size: Tokens per chunk.
        temperature: Divide logits by this before softmax (1.0 = no scaling).
        token_clip: Clamp per-element divergence to this value (0 = no clip).

    Returns:
        (loss, n_tokens) — scalar mean JSD loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    log_beta = math.log(beta) if beta > 0 else float("-inf")
    log_1m_beta = math.log(1.0 - beta) if beta < 1 else float("-inf")

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    jsd_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float() / temperature, dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float() / temperature, dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        # log(m) = logsumexp([log_p_T + log(beta), log_p_S + log(1-beta)])
        log_m = torch.logsumexp(
            torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0),
            dim=0,
        )

        kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True)
        del t_lp
        kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True)
        del s_lp, log_m

        jsd_per_vocab = beta * kl_t + (1.0 - beta) * kl_s
        del kl_t, kl_s
        if token_clip > 0:
            jsd_per_vocab = jsd_per_vocab.clamp(max=token_clip)
        jsd_sum = jsd_sum + jsd_per_vocab.sum(dim=-1).sum()
        del jsd_per_vocab

    return jsd_sum / n_tokens, n_tokens


def _compute_per_token_entropy_and_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    chunk_size: int = 512,
    jsd_topk: int = 0,
    jsd_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized per-token entropy and JSD in chunks.

    Args:
        student_logits: (N, V) student logits.
        teacher_logits: (N, V) teacher logits.
        chunk_size: Tokens per chunk.
        jsd_topk: If > 0, compute JSD only over teacher's top-K logits
            (renormalized). Reduces noise from the long tail of 152K vocab.
        jsd_scale: Multiplier on both normalized entropy and JSD to avoid
            numerical issues.

    Returns:
        (norm_entropy, norm_jsd) — both (N,).
        norm_entropy in [0, jsd_scale], norm_jsd in [0, jsd_scale].
    """
    n_tokens = student_logits.shape[0]
    ln_v = math.log(student_logits.shape[1])
    ln_2 = math.log(2.0)

    entropy_parts = []
    jsd_parts = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        s_chunk = student_logits[start:end].float()
        t_chunk = teacher_logits[start:end].float()

        # Normalized entropy (always full-vocab)
        s_lp_full = F.log_softmax(s_chunk, dim=-1)
        s_p_full = s_lp_full.exp()
        ent = -(s_p_full * s_lp_full).sum(dim=-1) / ln_v * jsd_scale
        entropy_parts.append(ent)
        del s_lp_full, s_p_full

        # JSD: top-K or full-vocab
        if jsd_topk > 0 and jsd_topk < s_chunk.shape[-1]:
            # Select teacher's top-K, gather both, renormalize over K
            _, topk_idx = t_chunk.topk(jsd_topk, dim=-1)
            t_topk = t_chunk.gather(-1, topk_idx)
            s_topk = s_chunk.gather(-1, topk_idx)
            t_lp = F.log_softmax(t_topk, dim=-1)
            s_lp = F.log_softmax(s_topk, dim=-1)
            t_p = t_lp.exp()
            s_p = s_lp.exp()
            del topk_idx, t_topk, s_topk
        else:
            s_lp = F.log_softmax(s_chunk, dim=-1)
            t_lp = F.log_softmax(t_chunk, dim=-1)
            s_p = s_lp.exp()
            t_p = t_lp.exp()

        m_p = 0.5 * (s_p + t_p)
        log_m = m_p.clamp(min=1e-8).log()
        jsd = 0.5 * (s_p * (s_lp - log_m)).sum(dim=-1) + \
              0.5 * (t_p * (t_lp - log_m)).sum(dim=-1)
        jsd_parts.append(jsd.clamp(min=0.0) / ln_2 * jsd_scale)

        del s_lp, s_p, t_lp, t_p, s_chunk, t_chunk, m_p, log_m, jsd

    return torch.cat(entropy_parts, dim=0), torch.cat(jsd_parts, dim=0)


def entropy_weighted_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sample_ratio: float,
    entropy_alpha: float,
    jsd_gamma: float = 0.0,
    jsd_topk: int = 0,
    jsd_scale: float = 1.0,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample tokens weighted by normalized entropy^alpha * jsd^gamma.

    Both entropy and JSD are normalized before exponentiation:
      - entropy_norm = entropy / ln(V)        where V = vocab size
      - jsd_norm     = jsd / ln(2) * jsd_scale

    weight(t) = entropy_norm(t)^alpha * jsd_norm(t)^gamma

    Args:
        student_logits: (N, V) student logits for response tokens.
        teacher_logits: (N, V) teacher logits for response tokens.
        sample_ratio: Fraction of tokens to keep (e.g. 0.5 = 50%).
        entropy_alpha: Exponent on normalized student entropy.
        jsd_gamma: Exponent on normalized JSD between student and teacher.
            0.0 disables JSD weighting (pure entropy sampling).
        jsd_topk: If > 0, compute JSD over teacher's top-K logits only.
        jsd_scale: Multiplier on normalized JSD (default 1.0).
        chunk_size: Chunk size for computation.

    Returns:
        (sampled_student_logits, sampled_teacher_logits, sample_info)
        sample_info dict contains per-token stats for logging:
          - norm_entropy: (N,) normalized entropy for all tokens
          - norm_jsd: (N,) normalized JSD for all tokens
          - sampled_indices: (K,) indices of sampled tokens
          - n_total: total token count before sampling
    """
    n_tokens = student_logits.shape[0]
    k = max(1, int(n_tokens * sample_ratio))
    if k >= n_tokens:
        return student_logits, teacher_logits, {}

    with torch.no_grad():
        norm_entropy, norm_jsd = _compute_per_token_entropy_and_jsd(
            student_logits, teacher_logits, chunk_size=chunk_size,
            jsd_topk=jsd_topk, jsd_scale=jsd_scale,
        )

        weights = norm_entropy.clamp(min=1e-8).pow(entropy_alpha)
        if jsd_gamma != 0.0:
            weights = weights * norm_jsd.clamp(min=1e-8).pow(jsd_gamma)

    # Sample without replacement
    indices = torch.multinomial(weights, num_samples=k, replacement=False)
    indices = indices.sort().values  # keep original order for chunk-wise loss

    sample_info = {
        "norm_entropy": norm_entropy,
        "norm_jsd": norm_jsd,
        "sampled_indices": indices,
        "n_total": n_tokens,
    }

    return student_logits[indices], teacher_logits[indices], sample_info


def compute_teacher_token_stats(
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token teacher entropy and teacher probability of the actual token.

    Used to distinguish two cases of high teacher entropy:
      A) Student went OOD — teacher entropy high AND p_T(y_t) low
      B) Genuine branch point — teacher entropy high AND p_T(y_t) normal

    Args:
        teacher_logits: (N, V) teacher logits at response positions.
        target_ids: (N,) actual next-token IDs at each response position.
        chunk_size: Chunk size for memory efficiency.

    Returns:
        (teacher_norm_entropy, teacher_token_prob) — both (N,)
        teacher_norm_entropy: H(p_T) / ln(V), normalized to [0, 1].
        teacher_token_prob: p_T(y_t) for the actual next token.
    """
    n_tokens = teacher_logits.shape[0]
    ln_v = math.log(teacher_logits.shape[1])

    entropy_parts = []
    prob_parts = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
        t_p = t_lp.exp()

        # Normalized entropy: H(p_T) / ln(V)
        ent = -(t_p * t_lp).sum(dim=-1) / ln_v
        entropy_parts.append(ent)

        # Teacher probability of actual token: p_T(y_t)
        chunk_target = target_ids[start:end]
        token_prob = t_p.gather(dim=-1, index=chunk_target.unsqueeze(-1)).squeeze(-1)
        prob_parts.append(token_prob)

        del t_lp, t_p

    return torch.cat(entropy_parts), torch.cat(prob_parts)


def compute_sampled_token_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    weight_mode: str = "reinforce",
    gate_beta: float = 5.0,
) -> tuple[torch.Tensor, int, dict]:
    """Sampled-token distillation loss (REINFORCE-style policy gradient).

    Only uses log-probabilities at the actually-sampled token, avoiding
    full-vocabulary computation. Three weight modes control the gradient signal:
      - reinforce: w = (log_t - log_s).detach()  (unbounded, original OPSD)
      - gated:    w = sigmoid(beta * (log_t - log_s))  (bounded [0,1], SDAR-style)
      - weighted: w = exp(log_s).detach() * (log_t - log_s)  (probability-weighted)

    loss = mean(-w * log_s)

    Args:
        student_log_probs: (N,) log π_student(y_t). Has gradients.
        teacher_log_probs: (N,) log π_teacher(y_t). Detached.
        weight_mode: "reinforce" | "gated" | "weighted"
        gate_beta: Sigmoid temperature (only for gated mode).

    Returns:
        (loss, n_tokens, metrics) — scalar loss, token count, diagnostics dict.
    """
    n_tokens = student_log_probs.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_log_probs.device, requires_grad=True), 0, {}

    teacher_log_probs = teacher_log_probs.detach()
    delta = teacher_log_probs - student_log_probs.detach()

    if weight_mode == "reinforce":
        weight = delta
    elif weight_mode == "gated":
        weight = torch.sigmoid(gate_beta * delta)
    elif weight_mode == "weighted":
        weight = student_log_probs.detach().exp() * delta
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    weight = weight.detach()
    loss = (-weight * student_log_probs).mean()

    with torch.no_grad():
        metrics = {
            "opd/sampled_weight_mean": weight.mean().item(),
            "opd/sampled_weight_pos_ratio": (weight > 0).float().mean().item(),
            "opd/sampled_delta_mean": delta.mean().item(),
        }

    return loss, n_tokens, metrics


def compute_topk_divergence_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    topk: int = 64,
    loss_type: str = "forward_kl",
    chunk_size: int = 512,
    temperature: float = 1.0,
    token_clip: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise divergence on teacher's top-k token set (renormalized).

    Computes KL/JSD only over teacher's top-k tokens per position, reducing
    noise from the long-tail of large vocabularies and saving compute.

    For each chunk:
      1. Select teacher's top-k logits and indices
      2. Gather student logits at those positions
      3. Renormalize both over k tokens (log_softmax over dim=-1)
      4. Compute divergence on the k-sized distribution

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        topk: Number of teacher top tokens to consider.
        loss_type: "forward_kl" | "reverse_kl" | "jsd"
        chunk_size: Tokens per chunk.
        temperature: Divide logits by this before renormalization.
        token_clip: Clamp per-element divergence (0 = no clip).

    Returns:
        (loss, n_tokens) — scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    div_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        # Get teacher top-k indices
        _, topk_idx = (t_chunk.float() / temperature).topk(topk, dim=-1)  # (chunk, k)

        # Gather both at teacher's top-k positions
        t_topk = t_chunk.float().gather(-1, topk_idx) / temperature  # (chunk, k)
        s_topk = student_logits[start:end].float().gather(-1, topk_idx) / temperature  # (chunk, k)
        del t_chunk
        teacher_chunks[i] = None

        # Renormalize over k tokens
        t_lp = F.log_softmax(t_topk, dim=-1)
        s_lp = F.log_softmax(s_topk, dim=-1)
        del t_topk, s_topk

        # Compute divergence
        if loss_type == "forward_kl":
            div_per_vocab = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
        elif loss_type == "reverse_kl":
            div_per_vocab = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
        elif loss_type == "jsd":
            log_m = torch.logsumexp(
                torch.stack([t_lp + math.log(0.5), s_lp + math.log(0.5)], dim=0), dim=0
            )
            kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True)
            kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True)
            div_per_vocab = 0.5 * kl_t + 0.5 * kl_s
            del log_m, kl_t, kl_s
        else:
            raise ValueError(f"Unknown loss_type for topk: {loss_type}")
        del t_lp, s_lp

        if token_clip > 0:
            div_per_vocab = div_per_vocab.clamp(max=token_clip)
        div_sum = div_sum + div_per_vocab.sum(dim=-1).sum()
        del div_per_vocab

    return div_sum / n_tokens, n_tokens


def compute_divergence_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    jsd_beta: float = 0.0,
    temperature: float = 1.0,
    token_clip: float = 0.0,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Unified divergence loss with continuous jsd_beta control.

    Provides a single entry point for all full-vocab divergence directions:
      - jsd_beta=0.0: Forward KL — KL(p_teacher || p_student), mean-seeking
      - jsd_beta=1.0: Reverse KL — KL(p_student || p_teacher), mode-seeking
      - 0 < jsd_beta < 1: JSD_beta(p_teacher || p_student), balanced

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        jsd_beta: 0=FKL, 1=RKL, (0,1)=JSD with this as interpolation weight.
        temperature: Divide logits by this before softmax.
        token_clip: Clamp per-element divergence (0 = no clip).
        chunk_size: Tokens per chunk.

    Returns:
        (loss, n_tokens) — scalar mean loss and token count.
    """
    if jsd_beta == 0.0:
        return compute_forward_kl_loss(
            teacher_logits, student_logits,
            chunk_size=chunk_size, temperature=temperature, token_clip=token_clip,
        )
    elif jsd_beta == 1.0:
        return compute_reverse_kl_loss(
            teacher_logits, student_logits,
            chunk_size=chunk_size, temperature=temperature, token_clip=token_clip,
        )
    else:
        return compute_jsd_loss(
            teacher_logits, student_logits,
            beta=jsd_beta, chunk_size=chunk_size, temperature=temperature, token_clip=token_clip,
        )


LOSS_FN_MAP = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
    "jsd": compute_jsd_loss,
}
