"""mini-swe-agent ↔ agent-sandbox-rl integration (gen-2).

Same SkyRL generator interface as :mod:`skyrl_sandbox.mini_swe_agent`, but sandbox provisioning is
delegated to an ``agent_sandbox_rl`` fleet: per-image SandboxTemplates + SandboxWarmPools sized to a
concurrency budget, one claim per rollout/eval, preflight, and run reports — instead of a cold
``Sandbox`` CR per trajectory. The env adapter (:class:`FleetSandboxEnvironment`) wraps an
already-claimed ``SandboxHandle``; fleet knobs live in the task YAML's ``environment:`` block
(:class:`Fleet2EnvironmentConfig`).
"""

from .environment import Fleet2EnvironmentConfig, FleetSandboxEnvironment

__all__ = ["Fleet2EnvironmentConfig", "FleetSandboxEnvironment"]
