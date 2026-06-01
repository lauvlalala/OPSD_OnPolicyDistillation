#!/bin/bash
# Train OPSD on OpenThoughts-30k (self-distillation with PI + EMA)
#
# Usage:
#   MODEL_PATH=/path/to/model bash scripts/opd/train_opsd_openthoughts.sh
#
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}

# Training
train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
learning_rate=${LEARNING_RATE:-1e-6}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
rollout_n=${ROLLOUT_N:-1}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}

# OPSD-specific
opd_loss_type=${OPD_LOSS_TYPE:-forward_kl}
opd_chunk_size=${OPD_CHUNK_SIZE:-256}
opd_max_length=${OPD_MAX_LENGTH:-16384}
opd_token_scope=${OPD_TOKEN_SCOPE:-full_vocab}
opd_temperature=${OPD_TEMPERATURE:-1.0}
opd_token_clip=${OPD_TOKEN_CLIP:-0.0}
teacher_sync_freq=${TEACHER_SYNC_FREQ:-1}
teacher_ema_decay=${TEACHER_EMA_DECAY:-0.999}

# Sampling
temperature=${TEMPERATURE:-1.0}
val_temperature=${VAL_TEMPERATURE:-0.6}
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ============================================================================
# Prepare data
# ============================================================================

DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data/openthoughts"}

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Preparing OpenThoughts data..."
    python3 "${SRC_ROOT}/data/prepare_openthoughts.py" \
        --output-dir "$DATA_DIR" \
        --num-samples 30000
fi

TRAIN_FILE="${DATA_DIR}/train.parquet"

# Validation: MATH-500 only (lightweight, frequent during training)
# Evaluation: MATH-500 + AIME 2024 + AIME 2025 (full benchmark, at end)
EVAL_DIR=${EVAL_DIR:-"${REPO_ROOT}/data/grpo_processed"}
VAL_MATH500="${EVAL_DIR}/val_math500.parquet"

# ============================================================================
# Launch
# ============================================================================

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
EXP_NAME=${MODEL_NAME_SAFE}-OPSD-openthoughts-${opd_loss_type}-ema${teacher_ema_decay}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== OPSD Training (OpenThoughts) ==="
echo "MODEL: $MODEL_PATH"
echo "LOSS: $opd_loss_type | SCOPE: $opd_token_scope | EMA: $teacher_ema_decay"
echo "OUTPUT: $output_dir"

python3 -m opd.main_opd \
    --config-path "${SRC_ROOT}/opd/config" \
    --config-name opd_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_MATH500']" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    opd.loss_type=${opd_loss_type} \
    opd.chunk_size=${opd_chunk_size} \
    opd.max_length=${opd_max_length} \
    opd.token_scope=${opd_token_scope} \
    opd.temperature=${opd_temperature} \
    opd.token_clip=${opd_token_clip} \
    opd.pi_mode=template \
    opd.teacher_sync_freq=${teacher_sync_freq} \
    opd.teacher_ema_decay=${teacher_ema_decay} \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/math_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs

echo "=== OPSD training (OpenThoughts) completed ==="
