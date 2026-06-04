#!/bin/bash
# Train OPSD on SWE-Gym Lite (multi-turn, bash tool in Docker)
#
# Prerequisites:
#   pip install docker
#   Docker daemon running
#   Pull SWE-Gym Lite images: docker pull xingyaoww/sweb.eval.x86_64.<instance_id>
#
# Usage:
#   MODEL_PATH=/path/to/model bash scripts/opd/train_opsd_swe_gym.sh
#
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
learning_rate=${LEARNING_RATE:-1e-6}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}
max_prompt_length=${MAX_PROMPT_LENGTH:-8000}
max_response_length=${MAX_RESPONSE_LENGTH:-32768}
rollout_n=${ROLLOUT_N:-8}
tp_size=${TP_SIZE:-2}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}
max_assistant_turns=${MAX_ASSISTANT_TURNS:-40}

opd_loss_type=${OPD_LOSS_TYPE:-jsd}
opd_chunk_size=${OPD_CHUNK_SIZE:-256}
opd_token_scope=${OPD_TOKEN_SCOPE:-full_vocab}
teacher_sync_freq=${TEACHER_SYNC_FREQ:-1}
teacher_ema_decay=${TEACHER_EMA_DECAY:-0.95}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
TOOL_CONFIG_PATH="${SRC_ROOT}/tools/swe/tool_config.yaml"

DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data/swe_gym"}

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Preparing SWE-Gym Lite data..."
    python3 "${SRC_ROOT}/data/prepare_swe_gym.py" \
        --output-dir "$DATA_DIR"
fi

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
EXP_NAME=${MODEL_NAME_SAFE}-OPSD-swe-gym-lite-rollout-ema${teacher_ema_decay}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== OPSD Training (SWE-Gym Lite, multi-turn bash) ==="
echo "MODEL: $MODEL_PATH | PI: rollout | TOOL: bash (docker)"

python3 -m opd.main_opd \
    --config-path "${SRC_ROOT}/opd/config" \
    --config-name opd_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_FILE']" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns \
    actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    opd.loss_type=${opd_loss_type} \
    opd.chunk_size=${opd_chunk_size} \
    opd.token_scope=${opd_token_scope} \
    opd.pi_mode=rollout \
    opd.teacher_sync_freq=${teacher_sync_freq} \
    opd.teacher_ema_decay=${teacher_ema_decay} \
    custom_reward_function.path="${SRC_ROOT}/rewards/swe_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True

echo "=== OPSD (SWE-Gym Lite) completed ==="
