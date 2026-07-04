#!/usr/bin/env bash
# Run the GEN-2 (agent-sandbox-rl) fleet smoke test FROM AN IN-CLUSTER POD, as the runner
# ServiceAccount — so it exercises the real RBAC + in-cluster-token path, including the NEW rules the
# fleet needs (template/pool create; CRD + runtimeclass reads for preflight). A laptop run uses your
# admin kubeconfig and bypasses RBAC.
#
# Prereqs: ./infra/up-smoke.sh has run (cluster + agent-sandbox + RBAC — re-run infra/05-setup-rbac.sh
# on clusters that predate mini_swe_agent_2), and kubectl points at it.
#
#   bash scripts/mini_swe_agent_2/run_smoke_in_pod.sh                  # defaults (gVisor on)
#   GVISOR= bash scripts/mini_swe_agent_2/run_smoke_in_pod.sh          # plain CPU pool
#   SMOKE_ARGS="--keep" bash scripts/mini_swe_agent_2/run_smoke_in_pod.sh   # leave pool up to inspect
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Namespaces from infra/.env (fall back to the documented defaults).
source "${REPO_DIR}/infra/load-config.sh" 2>/dev/null || true
RUNNER_NS="${RAY_NAMESPACE:-skyrl}"            # where the runner pod + SA live
SANDBOX_NS="${SANDBOX_NAMESPACE:-skyrl-sandboxes}"  # where the fleet creates pools/claims (RBAC-scoped here)
POD="${RUNNER_POD:-smoke-runner}"
# gVisor ON by default — up-smoke.sh provisions the gVisor pool. Set GVISOR= to disable (plain CPU pool).
GVISOR="${GVISOR:---gvisor}"

command -v kubectl >/dev/null || { echo "kubectl not found"; exit 1; }

# 1. Ensure the runner pod exists and is Ready.
if ! kubectl -n "$RUNNER_NS" get pod "$POD" >/dev/null 2>&1; then
  echo ">> applying runner pod ($RUNNER_NS/$POD)..."
  kubectl apply -f "${REPO_DIR}/infra/manifests/smoke-runner.yaml"
fi
echo ">> waiting for $RUNNER_NS/$POD to be Ready..."
kubectl -n "$RUNNER_NS" wait --for=condition=Ready "pod/$POD" --timeout=180s

# 2. Sync this repo into the pod (tar over `kubectl exec`; excludes .git/infra/venvs).
echo ">> syncing repo -> $POD:/workspace ..."
kubectl -n "$RUNNER_NS" exec "$POD" -- mkdir -p /workspace
tar czf - -C "$REPO_DIR" \
  --exclude='.git' --exclude='infra' --exclude='__pycache__' --exclude='.venv' --exclude='*.pyc' . \
  | kubectl -n "$RUNNER_NS" exec -i "$POD" -- tar xzf - -C /workspace

# 3. uv sync (core deps only — pulls agent-sandbox-rl from GitHub on first run; no GPU/skyrl) + run
#    the smoke test against the sandbox namespace. The pod's in-cluster token (the SA) authorizes
#    every template/pool/claim/exec call -> RBAC is real.
echo ">> running gen-2 fleet smoke test on $POD as SA (sandbox namespace=$SANDBOX_NS)..."
kubectl -n "$RUNNER_NS" exec -i "$POD" -- bash -lc "
  set -e
  cd /workspace
  uv sync
  uv run python scripts/mini_swe_agent_2/smoke_test_fleet.py --namespace '$SANDBOX_NS' $GVISOR ${SMOKE_ARGS:-}
"
