# skyrl-sandbox

Run [SkyRL](https://github.com/NovaSky-AI/SkyRL) RL workloads on the
[kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) backend (GKE) for environment rollouts. Each rollout occurs in a gvisor-isolated gke sandbox.

The repo contains three examples, one per package folder, that use **agent-sandbox** in different ways:


| | [`skyrl_sandbox/mini_swe_agent`](skyrl_sandbox/mini_swe_agent) | [`skyrl_sandbox/mini_swe_agent_2`](skyrl_sandbox/mini_swe_agent_2) | [`skyrl_sandbox/multiplication`](skyrl_sandbox/multiplication) |
|---|---|---|---|
| task | [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) SWE-bench | same | toy `a * b` |
| image | **per-instance** (thousands) | **per-instance** (thousands) | **one fixed** image |
| create | raw `Sandbox` CR (no template, cold start per trajectory) | [agent-sandbox-rl](https://github.com/kubernetes-sigs/agent-sandbox/tree/main/examples/agent-sandbox-rl) fleet: **warm pool per image**, sized to `max_concurrent` → claim per trajectory | SDK `create_sandbox(warmpool=…)` → pool → template |
| execute | Kubernetes **pod-exec** | Kubernetes **pod-exec** | SDK **`commands.run`** (in-image `:8888`) |
| needs `:8888` runtime image? | no | no | **yes** |

The two mini-swe-agent generations exist because of how warm pools work in agent-sandbox: a pool serves ONE
image, and mini-swe-agent has a separate docker image per instance, so gen-1 fell back to raw per-trajectory
`Sandbox` CRs (no pooling, cold image pull + pod start on every rollout). The GKE team's **agent-sandbox-rl**
package removes that limitation by managing a SandboxTemplate + SandboxWarmPool **per image**, sized to a
concurrency budget rather than task count — gen-2 delegates all sandbox orchestration to it, so a GRPO group
(`n_samples_per_prompt` rollouts of one instance = one image) draws warm pods from a shared pool, and gets
preflight/pre-pull/run-reports for free. Gen-1 is kept as the raw-CR reference implementation.

Across both, the Ray workers (driving the sandboxes) hold the Kubernetes identity/RBAC; the sandbox
pods run untrusted model code with **no** API token and gVisor isolation (`infra/05-setup-rbac.sh`).

## Install

```bash
# local dev — core deps only (env backends + dataset prep + smoke test; any platform, no GPU):
uv venv && uv pip install -e .          # uv selects Python 3.12 per requires-python
```
Training/generation pull the heavy `skyrl[fsdp]` stack (linux/GPU); the run scripts call
`uv run --extra fsdp` for you, so there's no separate install step there.

Cluster (shared by both examples): `cd infra && cp .env.example .env && $EDITOR .env` (set
`PROJECT_ID`), then `./up.sh` (GKE + KubeRay + agent-sandbox + gVisor pool + RBAC).

## Example 1 — mini-swe-agent (SWE-bench)

```bash
# data
uv run python -m skyrl_sandbox.mini_swe_agent.preprocess --output_dir ~/data/swe_gym_subset
# train (GPUs)
bash scripts/mini_swe_agent/run_mini_swe_agent_sandbox.sh
# OR generate-only against a remote endpoint (Qwen via Fireworks, litellm native provider; no GPUs):
FIREWORKS_AI_API_KEY=fw-... bash scripts/mini_swe_agent/run_generate_fireworks.sh
```

**Two LLM backends, one generator.** Training serves the policy on **your own vLLM** (H100s) via
SkyRL's in-process inference engine — an OpenAI-compatible HTTP endpoint reached with litellm's
`openai/` provider (`OPENAI_BASE_URL`, see [`.env.miniswe`](.env.miniswe)). The no-GPU generation demo
uses **Fireworks** via litellm's native `fireworks_ai/` provider. The generator switches between them
with `generator.miniswe_litellm_model_name`: empty → `openai/<model.path>` (local vLLM / training);
`fireworks_ai/…` → Fireworks (generation).

Backend selected by `environment_class:
"skyrl_sandbox.mini_swe_agent.environment.AgentSandboxEnvironment"` in
[`configs/mini_swe_agent/swebench_agent_sandbox.yaml`](configs/mini_swe_agent/swebench_agent_sandbox.yaml).
The example targets the mini-swe-agent **1.x** API, so `pyproject.toml` pins `mini-swe-agent<2`.

## Example 2 — mini-swe-agent gen-2 (SWE-bench, agent-sandbox-rl warm pools)

Same task, dataset, LLM suite, and generator interface as Example 1 — only the sandbox backend changes:
`MiniSweAgent2Generator` builds one `AsyncSandboxFleet` and lets `fleet.run(...)` warm per-image pools,
claim a sandbox per rollout (plus a **fresh** one per eval), and tear everything down each batch. Fleet
knobs (namespace, `max_concurrent`, `warmpool_strategy: sliding|naive|none`, gVisor placement) live in the
`environment:` block of
[`configs/mini_swe_agent_2/swebench_agent_sandbox_rl.yaml`](configs/mini_swe_agent_2/swebench_agent_sandbox_rl.yaml)
— consumed by this package, not by mini-swe-agent's env factory, since the fleet is shared across
trajectories. Rollouts run as asyncio tasks in the generator process (no per-trajectory Ray task: a fleet
holds kube clients + claim bookkeeping and can't be shipped to Ray workers).

```bash
# data: same parquet as Example 1 (skyrl_sandbox.mini_swe_agent.preprocess)
# train (GPUs)
bash scripts/mini_swe_agent_2/run_mini_swe_agent_sandbox.sh
# OR generate-only against Fireworks (no GPUs):
FIREWORKS_AI_API_KEY=fw-... bash scripts/mini_swe_agent_2/run_generate_fireworks.sh
```

`agent-sandbox-rl` isn't on PyPI; `pyproject.toml` pins it straight from the agent-sandbox repo
(`[tool.uv.sources]`, upstream PR #1000). **On clusters created before this example, re-run
`infra/05-setup-rbac.sh`**: the fleet additionally needs create/delete on SandboxTemplates/WarmPools and
cluster-scoped reads (CRDs, RuntimeClasses) for its preflight. Known deltas vs gen-1: template pods carry
resource **requests only** (no limits/ephemeral-storage — upstream `TemplateSpec` gap), and `cwd`/`env`
are folded into each exec instead of baked into the pod.

## Example 3 — multiplication (single image, agent-sandbox SDK)

```bash
# data
uv run python -m skyrl_sandbox.multiplication.dataset --output_dir ~/data/multiply_sandbox
# apply the SandboxTemplate + SandboxWarmPool -- FIRST set the template image to an agent-sandbox :8888
# runtime image (see caveat). 0.5.x spawns via the pool: claim -> SandboxWarmPool -> SandboxTemplate.
kubectl apply -f infra/manifests/sandbox-template-multiplication.yaml
kubectl apply -f infra/manifests/sandbox-warmpool-multiplication.yaml
# generate-only against Fireworks (no GPUs) -- same suite as mini-swe:
FIREWORKS_AI_API_KEY=fw-... bash scripts/multiplication/run_generate_fireworks.sh
# OR full GRPO training (GPUs), policy served by your own vLLM:
bash scripts/multiplication/run_multiply_sandbox.sh
```

Each trajectory adopts a `Sandbox` from the `multiplication-pool` warm pool (→ `multiplication-template`)
via `create_sandbox(warmpool=…)` and computes/verifies the product with the SDK's `commands.run`.
**Same LLM suite as mini-swe:** multiplication has a custom litellm-based generator
([`generator.py`](skyrl_sandbox/multiplication/generator.py), `MultiplyGenerator`), so the same
`*_litellm_model_name` decouple gives it both backends — **Fireworks** generation (`fireworks_ai/…`) and
**your-own-vLLM** training (`openai/<model.path>` + `OPENAI_BASE_URL`).

**Caveat:** the template image must ship the agent-sandbox `:8888` runtime server (left as a placeholder
in the manifest on purpose — there is no published default; build it from agent-sandbox's
`examples/python-runtime-sandbox/`).

## Testing the agent-sandbox part (no GPUs)

Validate the mini-swe sandbox contracts on a cheap **CPU** cluster — no H100s, no SkyRL training:
```bash
cd infra && ASSUME_YES=1 ./up-smoke.sh    # CPU cluster + gVisor sandbox pool + agent-sandbox + RBAC (no GPU/KubeRay)
cd .. && bash scripts/mini_swe_agent/run_smoke_in_pod.sh     # gen-1: raw Sandbox CR create → execute() → cleanup
bash scripts/mini_swe_agent_2/run_smoke_in_pod.sh            # gen-2: fleet preflight → warm pool → claim → execute() → fresh-box reclaim → teardown
# teardown:  ASSUME_YES=1 infra/teardown-smoke.sh
```
`run_smoke_in_pod.sh` runs the test from a pod as `skyrl-sandbox-runner`, so create → `execute()` →
cleanup is driven exactly as a SkyRL Ray worker would. Leave the Sandbox up for inspection with
`SMOKE_ARGS="--keep"`:
```bash
SMOKE_ARGS="--keep" bash scripts/mini_swe_agent/run_smoke_in_pod.sh
kubectl -n skyrl-sandboxes get sandboxes.agents.x-k8s.io -l app=mini-swe-agent-sandbox   # view
kubectl -n skyrl-sandboxes delete sandbox -l app=mini-swe-agent-sandbox                  # clean up
```
The per-phase detail, a laptop-only mode, and a `kubectl` fallback live in the headers of
[`smoke_test_agent_sandbox.py`](scripts/mini_swe_agent/smoke_test_agent_sandbox.py) and
[`run_smoke_in_pod.sh`](scripts/mini_swe_agent/run_smoke_in_pod.sh).
