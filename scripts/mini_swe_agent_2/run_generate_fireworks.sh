#!/usr/bin/env bash
# Generate-only run of the GEN-2 (agent-sandbox-rl) mini-swe-agent example against FIREWORKS (NO
# training, NO GPUs), via litellm's native fireworks_ai provider. The agent's bash runs in warm-pool
# sandboxes claimed from an agent-sandbox-rl fleet; Ray runs in-process (local mode).
#
# Prereqs:
#   - agent-sandbox cluster up (infra/up-smoke.sh is enough -- no GPU pool) + kubectl context set, OR
#     run this inside the runner pod (scripts/mini_swe_agent_2/run_generate_in_pod.sh) for real RBAC.
#     NOTE: the fleet path needs the EXTENDED RBAC (template/pool create + cluster reads) from the
#     current infra/05-setup-rbac.sh -- re-run it if your cluster predates mini_swe_agent_2.
#   - eval dataset (same data as gen-1):
#       uv run python -m skyrl_sandbox.mini_swe_agent.preprocess --output_dir "$DATA_DIR"
#   - FIREWORKS_AI_API_KEY exported (litellm's native env var for the fireworks_ai provider).
#
# TOKENIZER vs MODEL are DECOUPLED (via generator.miniswe_litellm_model_name -- see generator.py):
#   * TOKENIZER (= trainer.policy.model.path) must be a valid HF id; it only loads the tokenizer. Default Qwen/Qwen3-4B.
#   * FW_MODEL  (= the Fireworks model id) becomes the full litellm id `fireworks_ai/$FW_MODEL`.
#   The tokenizer is a STAND-IN for the served model -- fine for a generation test, NOT for training.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# litellm's fireworks_ai provider reads FIREWORKS_AI_API_KEY (accept FIREWORKS_API_KEY as an alias).
FIREWORKS_AI_API_KEY="${FIREWORKS_AI_API_KEY:-${FIREWORKS_API_KEY:-}}"
: "${FIREWORKS_AI_API_KEY:?set FIREWORKS_AI_API_KEY to your Fireworks key}"
export FIREWORKS_AI_API_KEY
# Optional litellm cost/context registry (not required for fireworks_ai; litellm has built-in data).
export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-configs/mini_swe_agent/litellm.json}"
# Use SkyRL's LEGACY inference path -- same rationale as gen-1's run_generate_fireworks.sh: the
# generator drives the LLM via litellm (Fireworks) and needs no SkyRL engine at all.
export _SKYRL_USE_NEW_INFERENCE=0
# Disable Ray's `uv run` runtime-env hook (see gen-1's script for the why).
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"   # HF id -> tokenizer (model.path)
# Fireworks model id -- VERIFY it exists in your catalog (https://fireworks.ai/models).
FW_MODEL="${FW_MODEL:-accounts/fireworks/models/gpt-oss-20b}"
DATA_DIR="${DATA_DIR:-$HOME/data/swe_gym_subset}"
CONFIG="${CONFIG:-$REPO_DIR/configs/mini_swe_agent_2/swebench_agent_sandbox_rl.yaml}"

# run_engines_locally=false -> no local vLLM/GPU; colocate_all=false -> no GPU placement group.
# eval-only knobs keep the batch tiny for a quick validation. Fleet knobs (max_concurrent, warm-pool
# strategy, namespace, gVisor placement) live in the CONFIG yaml's environment block.
uv run --extra "${SKYRL_EXTRA:-fsdp}" python -m skyrl_sandbox.mini_swe_agent_2.generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="$TOKENIZER" \
  trainer.max_prompt_length="${MAX_PROMPT:-8192}" \
  generator.miniswe_litellm_model_name="fireworks_ai/$FW_MODEL" \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.num_engines=0 \
  trainer.placement.colocate_all=false \
  trainer.eval_batch_size="${EVAL_BATCH_SIZE:-2}" \
  generator.n_samples_per_prompt=1 \
  generator.sampling_params.max_generate_length="${MAX_GEN:-2048}" \
  generator.sampling_params.logprobs=null \
  generator.max_input_length="${MAX_INPUT:-8192}" \
  generator.max_turns="${MAX_TURNS:-10}" \
  generator.miniswe_config_path="$CONFIG" \
  generator.miniswe_traj_dir="${MINISWE_TRAJ_DIR:-$HOME/mini_swe_agent_2_trajs_gen}" \
  trainer.logger="${LOGGER:-console}" \
  trainer.project_name=mini_swe_2_gen \
  trainer.run_name=mini_swe_2_fireworks_gen \
  "$@"
