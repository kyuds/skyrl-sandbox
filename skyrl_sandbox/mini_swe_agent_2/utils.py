"""Rollout-side helpers: env startup command + fleet-based trajectory evaluation.

Mirrors gen-1's ``utils.evaluate_trajectory`` semantics exactly — apply the model patch inline, run
the instance's eval script, ``resolved = returncode == 0`` — but the fresh evaluation sandbox is a
second **claim from the same warm pool** instead of a second cold ``Sandbox`` CR. Shared conventions
(image naming, the result TypedDict) are imported from gen-1 rather than duplicated.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict

from jinja2 import Template

from ..mini_swe_agent.utils import MiniSWEEvaluationResult
from .environment import Fleet2EnvironmentConfig, FleetSandboxEnvironment

if TYPE_CHECKING:
    from agent_sandbox_rl import AsyncSandboxFleet, SandboxHandle, Task

logger = logging.getLogger(__name__)


def make_env(fleet: "AsyncSandboxFleet", handle: "SandboxHandle", env_cfg: Fleet2EnvironmentConfig) -> FleetSandboxEnvironment:
    """Wrap a claimed handle in the mini-swe-agent env adapter (resolving its owning cluster)."""
    return FleetSandboxEnvironment(handle, fleet.registry.get(handle.cluster_name), env_cfg)


def run_env_startup_command(env: FleetSandboxEnvironment, sweagent_config: dict, instance: dict) -> None:
    """Run the optional ``run.env_startup_command`` (gen-1's get_sb_environment tail). Blocking."""
    if startup_command := sweagent_config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")


async def evaluate_trajectory_on_fleet(
    fleet: "AsyncSandboxFleet",
    task: "Task",
    instance: Dict[str, Any],
    model_patch: str,
    sweagent_config: dict,
    env_cfg: Fleet2EnvironmentConfig,
) -> MiniSWEEvaluationResult:
    """Grade one trajectory on a FRESH sandbox claimed from the task's warm pool.

    The agent's own sandbox is never reused (its repo state is mutated); callers release it before
    calling this, so peak sandbox demand stays ~max_concurrent instead of gen-1's 2x overlap. Blocking
    execs run in worker threads to keep the shared event loop free.
    """
    ret = MiniSWEEvaluationResult(instance_id=instance["instance_id"], resolved=False, eval_error=None)

    try:
        handle = await fleet.acquire(task)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info("Starting eval environment failed for %s: %s", instance["instance_id"], e, exc_info=True)
        return ret

    try:
        env = make_env(fleet, handle, env_cfg)
        await asyncio.to_thread(run_env_startup_command, env, sweagent_config, instance)

        # Apply the git patch in-line (bounded by ARG_MAX ~1MB; larger patches are meant to fail),
        # then run the eval script — identical contract to gen-1.
        delimiter = f"PATCH_{uuid.uuid4().hex}"  # unlikely to collide with symbols in the patch
        command = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"
        obs = await asyncio.to_thread(env.execute, command)

        if obs["returncode"] != 0:
            ret["eval_error"] = obs["output"]
        else:
            eval_script = instance["eval_script"]
            eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
            obs = await asyncio.to_thread(lambda: env.execute(eval_cmd, timeout=3600))
            ret["resolved"] = obs["returncode"] == 0
            ret["eval_error"] = (
                f"(truncated to last 1000 characters)\n{obs['output'][-1000:]}" if not ret["resolved"] else None
            )
        return ret
    finally:
        # Hand the pod back to the fleet (claim delete + bookkeeping); never let this mask the result.
        try:
            await fleet.release(handle)
        except Exception:
            logger.warning("failed to release eval sandbox for %s", instance["instance_id"], exc_info=True)
