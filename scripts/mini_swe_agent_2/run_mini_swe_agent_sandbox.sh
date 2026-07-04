#!/usr/bin/env bash
set -x

# Standalone launcher: GRPO training (Qwen3-8B, 1x8 H100) on the SWE-bench task -- GEN-2 variant:
# same SkyRL training stack as scripts/mini_swe_agent/run_mini_swe_agent_sandbox.sh, but rollout/eval
# sandboxes come from an agent-sandbox-rl warm-pool fleet (per-image pools sized to max_concurrent in
# the config yaml) instead of a cold Sandbox CR per trajectory.
#
# Prerequisites:
#   1. Cluster up:  (cd infra && ./up.sh)     # GKE + KubeRay + agent-sandbox + gVisor pool + RBAC
#      (re-run infra/05-setup-rbac.sh on older clusters -- the fleet needs the extended RBAC)
#   2. Deps:        uv sync --extra fsdp       # skyrl[fsdp] + this package + agent-sandbox-rl (linux/GPU)
#   3. Data:        uv run python -m skyrl_sandbox.mini_swe_agent.preprocess --output_dir "$DATA_DIR"
#   4. Ray worker pods run as the SANDBOX_RUNNER_SA identity (infra/05-setup-rbac.sh).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

CONFIG="${CONFIG:-$REPO_DIR/configs/mini_swe_agent_2/swebench_agent_sandbox_rl.yaml}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env.miniswe}"
DATA_DIR="${DATA_DIR:-$HOME/data/swe_gym_subset}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/llm_mini_swe_2}"
MINISWE_TRAJ_DIR="${MINISWE_TRAJ_DIR:-$HOME/mini_swe_agent_2_trajs}"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
LOGGER=wandb

uv run --extra fsdp --env-file "$ENV_FILE" python -m skyrl_sandbox.mini_swe_agent_2.main \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=2 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=30720 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=True \
  generator.inference_engine.enable_http_endpoint=True \
  generator.inference_engine.http_endpoint_host='127.0.0.1' \
  generator.inference_engine.http_endpoint_port=8001 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe_2" \
  trainer.run_name="mini_swe_2_8B_agent_sandbox_rl" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.miniswe_config_path="$CONFIG" \
  generator.miniswe_traj_dir="$MINISWE_TRAJ_DIR" \
  "$@"
