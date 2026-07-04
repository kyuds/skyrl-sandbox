"""A mini-swe-agent ``Environment`` backed by an **agent-sandbox-rl** warm-pool sandbox.

Gen-2 counterpart of :mod:`skyrl_sandbox.mini_swe_agent.environment`. Gen-1 creates a cold ``Sandbox``
CR per trajectory (per-instance SWE-bench images defeat single-template warm pools); here the
per-image SandboxTemplate/SandboxWarmPool/claim lifecycle is owned by an ``agent_sandbox_rl`` fleet,
and this class merely adapts one **already-claimed** :class:`~agent_sandbox_rl.SandboxHandle` to the
mini-swe-agent ``Environment`` protocol (``execute``/``cleanup``/``get_template_vars``/``serialize``).

Two deliberate differences from gen-1:

* **Not constructed via mini-swe-agent's dotted-path env factory.** A fleet (warm pools, claim
  bookkeeping) must be shared across trajectories, so the generator builds it and injects a handle
  per trajectory — same contract as agent-sandbox-rl's R2E-Gym adapter (``make_fleet_repo_env``).
  The task YAML's ``environment:`` block still holds all knobs (:class:`Fleet2EnvironmentConfig`);
  it is parsed by this package, not by ``minisweagent.get_environment``.
* **``cleanup()`` is a no-op.** The fleet owns the pod: ``fleet.release(handle)`` deletes the claim.
  An env must never delete a sandbox out from under the fleet's bookkeeping.

Exec goes through the Kubernetes **pod-exec** API with the exit code recovered from the exec
ERROR_CHANNEL — NOT ``SandboxHandle.exec``, which returns only merged output (no returncode, no
timeout). The RL reward is the eval script's ``returncode == 0``, so this path never silently maps a
failure to 0 (same contract as gen-1's ``kubernetes_util``).

**Security parity with gen-1:** the sandbox pod runs untrusted model-generated bash, so the fleet
template gives it **no** API token (``automountServiceAccountToken: false`` via ``extra_pod_spec``)
and pins it to the gVisor pool. The Kubernetes identity (RBAC) belongs to the process driving the
fleet, not the sandbox.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from kubernetes.client.rest import ApiException
from kubernetes.stream import stream as k8s_stream
from kubernetes.stream.ws_client import ERROR_CHANNEL

if TYPE_CHECKING:
    from agent_sandbox_rl import Cluster, SandboxHandle


@dataclass
class Fleet2EnvironmentConfig:
    """Schema of the ``environment:`` block in the gen-2 task YAML.

    One flat block (gen-1 parity) covering both halves of the integration: how each command runs
    inside a sandbox pod (exec adapter), and how the fleet provisions those pods (cluster targeting,
    orchestration, template shape). ``fleet_util.build_fleet`` consumes the fleet fields.
    """

    # --- exec adapter (per-command behavior inside the sandbox pod) ----------------------------
    cwd: str = "/testbed"
    """Working directory. The fleet's SandboxTemplate has no ``workingDir`` (unlike gen-1's pod), so
    every exec is prefixed with ``cd <cwd> &&``."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables, exported inline on every exec (gen-1 baked them into the pod ``env``;
    the fleet template has no env hook)."""
    forward_env: list[str] = field(default_factory=list)
    """Host (driver) env vars to forward; exported inline per-exec when set on the host."""
    timeout: int = 180
    """Default per-command timeout in seconds (eval passes a longer one explicitly)."""
    container_name: str = "agent-runtime"
    """Exec target container. Fixed to ``agent-runtime`` by agent-sandbox-rl's template renderer."""

    # --- fleet: cluster targeting ---------------------------------------------------------------
    namespace: str = "skyrl-sandboxes"
    """Namespace for templates/pools/claims/pods (matches SANDBOX_NAMESPACE in infra/.env)."""
    context: Optional[str] = None
    """Kube context name (None = current)."""
    kubeconfig: Optional[str] = None
    """Kubeconfig path (None = default)."""
    in_cluster: Optional[bool] = None
    """Use the in-cluster ServiceAccount. None = auto-detect from the mounted SA token."""

    # --- fleet: orchestration -------------------------------------------------------------------
    max_concurrent: int = 8
    """THE cost/throughput knob: sizes warm pools AND bounds parallel claim+rollout."""
    max_warmpool_size: int = 8
    """Hard cap on replicas per image pool."""
    warmpool_strategy: str = "sliding"
    """When pools exist: ``naive`` (all up front) | ``sliding`` (rolling window) | ``none``."""
    window_size: Optional[int] = None
    """Sliding window size in images. None = auto from max_concurrent."""
    ready_timeout: int = 600
    """Seconds to wait for pool/claim readiness (SWE-bench images are large: pull + schedule)."""

    # --- fleet: template / pod shape (requests only — TemplateSpec has no limits knob) -----------
    cpu: str = "1"
    memory: str = "2Gi"
    runtime_class: Optional[str] = "gvisor"
    """gVisor RuntimeClass for kernel isolation of untrusted bash. None to disable."""
    node_selector: dict[str, str] = field(default_factory=lambda: {"sandbox.gke.io/runtime": "gvisor"})
    """Pin sandbox pods to the GKE Sandbox (gVisor) node pool."""
    tolerations: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"key": "sandbox.gke.io/runtime", "operator": "Equal", "value": "gvisor", "effect": "NoSchedule"}
        ]
    )
    """Tolerate the gVisor node-pool taint (merged into the template via ``extra_pod_spec``)."""
    image_pull_secret: Optional[str] = None
    automount_service_account_token: bool = False
    """SECURITY: untrusted model-generated code must NOT receive a Kubernetes API token."""
    labels: dict[str, str] = field(default_factory=dict)
    """Extra labels merged onto every fleet-created resource (``app=agent-sandbox-rl`` always kept —
    it is the fleet's teardown/GC selector)."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], logger: logging.Logger | None = None) -> "Fleet2EnvironmentConfig":
        """Build from the YAML block, ignoring unknown keys (so an unrelated field can't crash us)."""
        known = {f.name for f in dataclasses.fields(cls)}
        ignored = set(raw) - known - {"environment_class"}  # tolerate a stale gen-1 key
        if ignored and logger:
            logger.warning("Fleet2EnvironmentConfig ignoring unknown config keys: %s", sorted(ignored))
        return cls(**{k: v for k, v in raw.items() if k in known})


def _parse_exec_returncode(error_channel: str) -> int:
    """Recover the command exit code from a pod-exec ERROR_CHANNEL status message.

    The Kubernetes exec API reports completion as a v1.Status JSON on the error channel:
    ``{"status": "Success"}`` (rc 0) or ``{"status": "Failure", "details": {"causes":
    [{"reason": "ExitCode", "message": "<n>"}]}}``. A failed/garbled status maps to nonzero (never a
    false 0) so an eval can't be silently marked resolved; an empty channel maps to 0 (the
    quiet-success case the k8s client itself assumes).
    """
    if not error_channel:
        return 0
    try:
        status = json.loads(error_channel)
    except (ValueError, TypeError):
        return 1
    if status.get("status") == "Success":
        return 0
    for cause in (status.get("details") or {}).get("causes", []):
        if cause.get("reason") == "ExitCode":
            try:
                return int(cause.get("message"))
            except (TypeError, ValueError):
                return 1
    return 1


def exec_in_pod_with_returncode(
    core_api, pod_name: str, namespace: str, container: str, argv: list[str], timeout: int
) -> dict[str, Any]:
    """Run ``argv`` in a pod via the Kubernetes pod-exec API, with timeout + real exit code.

    Returns ``{"output": <merged stdout+stderr>, "returncode": <int>}`` (``-1`` on timeout/transport
    error). This is the piece ``SandboxHandle.exec`` doesn't provide (it preloads content: no rc, no
    timeout); ``core_api`` must be a per-thread client — the kubernetes websocket ``stream()`` is not
    thread-safe across a shared one (use ``Cluster.exec_core_api()``).
    """
    try:
        resp = k8s_stream(
            core_api.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            command=argv,
            container=container,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
    except ApiException as e:
        return {"output": f"pod exec failed to start: {e}", "returncode": -1}

    chunks: list[str] = []
    start = time.monotonic()
    error_channel = ""
    try:
        while resp.is_open():
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return {"output": "".join(chunks) + f"\n<command timed out after {timeout}s>", "returncode": -1}
            resp.update(timeout=min(1.0, max(0.1, timeout - elapsed)))
            if resp.peek_stdout():
                chunks.append(resp.read_stdout())
            if resp.peek_stderr():
                chunks.append(resp.read_stderr())
        error_channel = resp.read_channel(ERROR_CHANNEL)
    finally:
        resp.close()

    return {"output": "".join(chunks), "returncode": _parse_exec_returncode(error_channel)}


class FleetSandboxEnvironment:
    """mini-swe-agent ``Environment`` over one claimed agent-sandbox-rl ``SandboxHandle``.

    One env per handle, one handle per rollout/eval phase. The env only *uses* the pod; acquisition
    and release stay with the fleet (see module docstring).
    """

    def __init__(
        self,
        handle: "SandboxHandle",
        cluster: "Cluster",
        config: Fleet2EnvironmentConfig,
        logger: logging.Logger | None = None,
    ):
        self.handle = handle
        self._cluster = cluster
        self.config = config
        self.logger = logger or logging.getLogger("skyrl_sandbox.mini_swe_agent_2.environment")

    # --- execution ---------------------------------------------------------------------------

    def execute(self, command: Any, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a bash command in the sandbox pod.

        Accepts a plain string (mini-swe-agent 1.x) or an ``{"command": ...}`` dict (2.x).
        Returns ``{"output": <merged stdout+stderr>, "returncode": <int>}`` (``-1`` on timeout/error).
        """
        cmd_str = command.get("command", "") if isinstance(command, dict) else command

        # Unlike gen-1, the pod carries no workingDir/env (the fleet template can't set them), so
        # both are folded into every exec: exports first, then cd, then the command.
        prelude = ""
        for key, value in self.config.env.items():
            prelude += f"export {key}={shlex.quote(str(value))}; "
        for key in self.config.forward_env:
            value = os.getenv(key)
            if value is not None:
                prelude += f"export {key}={shlex.quote(value)}; "
        workdir = cwd or self.config.cwd
        if workdir:
            prelude += f"cd {shlex.quote(workdir)} && "
        script = f"{prelude}{cmd_str}" if prelude else cmd_str

        # exec_core_api() is thread-local: rollouts run in worker threads, one websocket client each.
        return exec_in_pod_with_returncode(
            self._cluster.exec_core_api(),
            self.handle.pod_name,
            self._cluster.namespace,
            self.config.container_name,
            ["bash", "-lc", script],
            timeout or self.config.timeout,
        )

    # --- lifecycle -----------------------------------------------------------------------------

    def cleanup(self) -> None:
        """No-op: the FLEET owns the pod (``fleet.release(handle)`` deletes the claim)."""

    # --- introspection ---------------------------------------------------------------------------

    @property
    def pod_name(self) -> str:
        """Backing pod name (the exec target)."""
        return self.handle.pod_name

    @property
    def hostname(self) -> str:
        """The sandbox's stable in-cluster DNS name."""
        return self.handle.hostname

    # --- misc (mini-swe-agent Environment protocol) ------------------------------------------

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": asdict(self.config),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
