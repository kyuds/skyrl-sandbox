#!/usr/bin/env bash
# Run the GEN-2 (agent-sandbox-rl) generate-only example (Fireworks; NO GPUs, NO training) FROM AN
# IN-CLUSTER POD, as the runner ServiceAccount — the fleet analogue of gen-1's run_generate_in_pod.sh.
# The pod is the trusted driver: it creates SandboxTemplates/WarmPools, claims sandboxes, and execs
# into them through real RBAC, while the LLM is reached via litellm -> Fireworks. (The untrusted
# sandbox pods never get the API key.)
#
# What it does (idempotent; re-runnable):
#   1. ensure the `fireworks` Secret exists (from $FIREWORKS_AI_API_KEY if missing)
#   2. ensure the generate-runner pod exists + is Ready (infra/manifests/generate-runner.yaml)
#   3. tar this repo into the pod (excludes .git/infra/venvs, so the installed .venv is preserved)
#   4. in the pod: `uv sync --extra train` (fetches agent-sandbox-rl from GitHub on first run),
#      preprocess a small dataset (gen-1's preprocess -- same data), then run
#      scripts/mini_swe_agent_2/run_generate_fireworks.sh
#
# Prereqs: a cluster with agent-sandbox + the EXTENDED RBAC up (re-run infra/05-setup-rbac.sh if the
# cluster predates mini_swe_agent_2 -- the fleet needs template/pool create + CRD/runtimeclass reads),
# kubectl pointing at it, and a Fireworks key (export FIREWORKS_AI_API_KEY, or pre-create the Secret).
#
# Usage:
#   FIREWORKS_AI_API_KEY=fw-... bash scripts/mini_swe_agent_2/run_generate_in_pod.sh
#   # smaller/bigger runs and model overrides (all optional, shown with defaults):
#   FW_MODEL=accounts/fireworks/models/gpt-oss-20b DATA_LIMIT=2 EVAL_BATCH_SIZE=2 SKYRL_EXTRA=train \
#     bash scripts/mini_swe_agent_2/run_generate_in_pod.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Namespaces from infra/.env (fall back to the documented defaults).
source "${REPO_DIR}/infra/load-config.sh" 2>/dev/null || true
RUNNER_NS="${RAY_NAMESPACE:-skyrl}"                  # where the runner pod + SA live
SANDBOX_NS="${SANDBOX_NAMESPACE:-skyrl-sandboxes}"   # where the fleet creates pools/claims (RBAC-scoped here)
POD="${RUNNER_POD:-generate-runner}"
SECRET_NAME="${SECRET_NAME:-fireworks}"

# Generation knobs (forwarded into the pod; defaults match the gen-1 validated smoke run).
SKYRL_EXTRA="${SKYRL_EXTRA:-train}"                  # skyrl[skyrl-train] (no vLLM/GPU). 'fsdp' for the full stack.
FW_MODEL="${FW_MODEL:-accounts/fireworks/models/gpt-oss-20b}"   # must exist in your Fireworks catalog
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
DATA_LIMIT="${DATA_LIMIT:-2}"                        # rows per split in the in-pod dataset (preprocess --limit)
DATA_DIR_IN_POD="${DATA_DIR_IN_POD:-/root/data/swe_gym_subset}"

command -v kubectl >/dev/null || { echo "kubectl not found"; exit 1; }

# 1. Resolve the Fireworks key (host env wins; else read the existing Secret) and ensure the Secret
#    exists for future runs. The key is later piped over stdin (step 4) — never placed in argv.
FIREWORKS_AI_API_KEY="${FIREWORKS_AI_API_KEY:-${FIREWORKS_API_KEY:-}}"
if [ -n "$FIREWORKS_AI_API_KEY" ]; then
  if ! kubectl -n "$RUNNER_NS" get secret "$SECRET_NAME" >/dev/null 2>&1; then
    echo ">> creating Secret $RUNNER_NS/$SECRET_NAME from FIREWORKS_AI_API_KEY"
    kubectl -n "$RUNNER_NS" create secret generic "$SECRET_NAME" \
      --from-literal=FIREWORKS_AI_API_KEY="$FIREWORKS_AI_API_KEY" >/dev/null
  fi
elif kubectl -n "$RUNNER_NS" get secret "$SECRET_NAME" >/dev/null 2>&1; then
  FIREWORKS_AI_API_KEY="$(kubectl -n "$RUNNER_NS" get secret "$SECRET_NAME" \
    -o go-template='{{index .data "FIREWORKS_AI_API_KEY" | base64decode}}')"
else
  echo "ERROR: no Fireworks key. Export FIREWORKS_AI_API_KEY, or create the '$SECRET_NAME' secret in ns '$RUNNER_NS'." >&2
  exit 1
fi

# 2. Ensure the runner pod exists and is Ready.
if ! kubectl -n "$RUNNER_NS" get pod "$POD" >/dev/null 2>&1; then
  echo ">> applying runner pod ($RUNNER_NS/$POD)..."
  kubectl apply -f "${REPO_DIR}/infra/manifests/generate-runner.yaml"
fi
echo ">> waiting for $RUNNER_NS/$POD to be Ready..."
kubectl -n "$RUNNER_NS" wait --for=condition=Ready "pod/$POD" --timeout=300s

# 3. Sync this repo into the pod (tar over `kubectl exec`; excludes .git/infra/venvs so the installed
#    .venv is preserved across re-runs).
echo ">> syncing repo -> $POD:/workspace ..."
kubectl -n "$RUNNER_NS" exec "$POD" -- mkdir -p /workspace
tar czf - -C "$REPO_DIR" \
  --exclude='.git' --exclude='infra' --exclude='__pycache__' --exclude='.venv' --exclude='*.pyc' . \
  | kubectl -n "$RUNNER_NS" exec -i "$POD" -- tar xzf - -C /workspace

# 4. In the pod: install (cached after first run) -> small dataset -> generate. The key is read from
#    stdin (first line) so it never appears in the process args. Config values are interpolated by THIS
#    shell; $FW / $FIREWORKS_AI_API_KEY are escaped so they expand inside the pod.
echo ">> running gen-2 generate on $POD (model=$FW_MODEL, extra=$SKYRL_EXTRA, batch=$EVAL_BATCH_SIZE, limit=$DATA_LIMIT)..."
printf '%s\n' "$FIREWORKS_AI_API_KEY" | kubectl -n "$RUNNER_NS" exec -i "$POD" -- bash -lc "
  set -e
  read -r FW; export FIREWORKS_AI_API_KEY=\"\$FW\"
  cd /workspace
  echo '== [1/3] install (skyrl[$SKYRL_EXTRA] + agent-sandbox-rl) =='
  uv sync --extra '$SKYRL_EXTRA'
  echo '== [2/3] preprocess (--limit $DATA_LIMIT; gen-1 preprocess, same dataset) =='
  uv run --no-sync python -m skyrl_sandbox.mini_swe_agent.preprocess --output_dir '$DATA_DIR_IN_POD' --limit '$DATA_LIMIT'
  echo '== [3/3] generate (Fireworks: $FW_MODEL, agent-sandbox-rl fleet) =='
  SKYRL_EXTRA='$SKYRL_EXTRA' FW_MODEL='$FW_MODEL' EVAL_BATCH_SIZE='$EVAL_BATCH_SIZE' DATA_DIR='$DATA_DIR_IN_POD' \
    bash scripts/mini_swe_agent_2/run_generate_fireworks.sh 2>&1 | tee /tmp/generate2.log
"

echo ">> done. Trajectories are in the pod under \$HOME/mini_swe_agent_2_trajs_gen."
echo "   pull them:  kubectl -n $RUNNER_NS cp $POD:/root/mini_swe_agent_2_trajs_gen ./mini_swe_agent_2_trajs_gen"
echo "   leftovers:  kubectl -n $SANDBOX_NS get sandboxtemplates,sandboxwarmpools,sandboxclaims,sandboxes -l app=agent-sandbox-rl   # should be empty (fleet teardown)"
