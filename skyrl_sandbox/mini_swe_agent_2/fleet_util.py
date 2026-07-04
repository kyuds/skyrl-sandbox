"""Config → fleet wiring for the gen-2 example.

Everything that touches the ``agent_sandbox_rl`` orchestration API (building the fleet from the task
YAML's ``environment:`` block, mapping SkyRL instances to fleet ``Task``s) is confined here, so
``generator.py`` reads as the rollout logic and ``environment.py`` as the exec adapter.
"""

from __future__ import annotations

import os

from agent_sandbox_rl import (
    AsyncSandboxFleet,
    ClusterConfig,
    FleetConfig,
    ResourceSpec,
    Task,
    TemplateSpec,
    constants,
)

from ..mini_swe_agent.utils import get_docker_image_name
from .environment import Fleet2EnvironmentConfig

# The strategies AsyncSandboxFleet.run accepts (validated up front: a typo would otherwise surface
# only after warm pools were already provisioned for the batch).
WARMPOOL_STRATEGIES = ("naive", "sliding", "none")

_IN_CLUSTER_TOKEN = "/var/run/secrets/kubernetes.io/serviceaccount/token"


def detect_in_cluster() -> bool:
    """True when running inside a pod with a mounted ServiceAccount token."""
    return os.path.exists(_IN_CLUSTER_TOKEN)


def build_fleet(cfg: Fleet2EnvironmentConfig) -> AsyncSandboxFleet:
    """One single-cluster ``AsyncSandboxFleet`` from the YAML ``environment:`` block."""
    if cfg.warmpool_strategy not in WARMPOOL_STRATEGIES:
        raise ValueError(
            f"environment.warmpool_strategy must be one of {WARMPOOL_STRATEGIES}, got {cfg.warmpool_strategy!r}"
        )

    # TemplateSpec has no first-class automount/tolerations knobs; both merge at the POD level via
    # extra_pod_spec. (Container-level fields — limits, env, workingDir — are NOT reachable this way:
    # a `containers` key would replace the image-bearing container. See the design doc.)
    extra_pod_spec: dict = {"automountServiceAccountToken": cfg.automount_service_account_token}
    if cfg.tolerations:
        extra_pod_spec["tolerations"] = [dict(t) for t in cfg.tolerations]

    template = TemplateSpec(
        resources=ResourceSpec(cpu=cfg.cpu, memory=cfg.memory),
        runtime_class=cfg.runtime_class,
        node_selector=dict(cfg.node_selector) if cfg.node_selector else None,
        image_pull_secret=cfg.image_pull_secret,
        extra_pod_spec=extra_pod_spec,
    )

    cluster = ClusterConfig(
        name="skyrl",
        namespace=cfg.namespace,
        context=cfg.context,
        kubeconfig=cfg.kubeconfig,
        in_cluster=cfg.in_cluster if cfg.in_cluster is not None else detect_in_cluster(),
    )

    fleet_config = FleetConfig(
        clusters=[cluster],
        max_concurrent=cfg.max_concurrent,
        max_warmpool_size=cfg.max_warmpool_size,
        window_size=cfg.window_size,
        ready_timeout=cfg.ready_timeout,
        template=template,
        template_name_prefix="mswe2-img-",
        # app=agent-sandbox-rl must survive any custom labels: it is the fleet's teardown selector.
        labels={**cfg.labels, **dict(constants.DEFAULT_LABELS)},
    )
    return AsyncSandboxFleet(fleet_config)


def build_task(instance: dict, data_source: str, repetition_id, index: int) -> Task:
    """Map one SkyRL trajectory to a fleet ``Task``.

    The image reuses gen-1's SWE-bench/SWE-Gym naming convention (single source of truth). Tasks
    carry only the batch ``index``; the heavyweight per-trajectory context (instance dict, sampling
    params, ...) stays in the generator's closure, looked up by that index. The id is
    ``<instance>-<repetition>-<index>`` — unique even though a GRPO group repeats the same instance
    (and, on purpose, the same image: that is what the warm pool amortizes).
    """
    image = get_docker_image_name(instance, data_source)
    return Task(
        id=f"{instance['instance_id']}-{repetition_id}-{index}",
        image=image,
        metadata={"index": index},
    )
