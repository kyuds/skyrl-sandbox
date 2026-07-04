"""Smoke test for the GEN-2 (agent-sandbox-rl) sandbox path against a live (CPU) cluster.

Proves the fleet integration works **without SkyRL or GPUs**. It drives the exact pieces the gen-2
generator uses — ``Fleet2EnvironmentConfig.from_dict`` → ``build_fleet`` → warm pool → ``acquire`` →
``FleetSandboxEnvironment.execute(cmd) -> {output, returncode}`` → release → fresh-box re-acquire
(the eval pattern) → ``teardown`` — so a green run here is direct evidence the SkyRL ↔
agent-sandbox-rl path will work.

It can run from your laptop (kubeconfig) or in-cluster (run_smoke_in_pod.sh; real RBAC). Cluster
requirements: the agent-sandbox controller + v1beta1 CRDs (infra steps 01 + 04) and, in-cluster, the
EXTENDED runner RBAC from infra/05-setup-rbac.sh (template/pool create + CRD/runtimeclass reads).

    uv run python scripts/mini_swe_agent_2/smoke_test_fleet.py --namespace default

Useful flags:
    --image IMG          container image (default python:3.11-slim; any image with bash works)
    --namespace NS       namespace for pools/claims (default skyrl-sandboxes; use one you can write to)
    --cwd DIR            working dir for execs (default /tmp; SWE-bench images use /testbed)
    --gvisor             pin to the gVisor pool (only if infra step 02 sandbox pool exists)
    --keep               skip release/teardown; leave the pool + claimed sandbox up for inspection
"""

import argparse
import asyncio
import sys

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


async def run(args) -> int:
    from skyrl_sandbox.mini_swe_agent_2.environment import Fleet2EnvironmentConfig
    from skyrl_sandbox.mini_swe_agent_2.fleet_util import build_fleet
    from skyrl_sandbox.mini_swe_agent_2.utils import make_env
    from agent_sandbox_rl import Task

    # gVisor OFF by default so the pod schedules on any vanilla CPU node; modest resources so it fits
    # small nodes / Autopilot. Built via from_dict — the same entry point the generator uses.
    isolation = (
        {"runtime_class": "gvisor"}
        if args.gvisor
        else {"runtime_class": None, "node_selector": {}, "tolerations": []}
    )
    env_cfg = Fleet2EnvironmentConfig.from_dict(
        {
            "cwd": args.cwd,
            "timeout": 60,
            "env": {"SMOKE_VAR": "hello-env"},
            "namespace": args.namespace,
            "max_concurrent": 2,
            "max_warmpool_size": 2,
            "warmpool_strategy": "naive",
            "ready_timeout": 300,
            "cpu": "100m",
            "memory": "128Mi",
            "automount_service_account_token": False,
            **isolation,
        }
    )
    fleet = build_fleet(env_cfg)
    task = Task(id="smoke-0", image=args.image, metadata={"index": 0})
    fleet.load_tasks([task])

    print(f"\n== Phase 1: preflight + plan + warm pool (ns={args.namespace}, image={args.image}) ==")
    try:
        await fleet.setup()
    except Exception as e:  # give actionable hints for the common failures
        print(f"\nFAILED to set up the fleet: {type(e).__name__}: {e}\n")
        msg = str(e).lower()
        if "preflight" in msg and ("crd" in msg or "not found" in msg):
            print("  → Are the agent-sandbox controller + v1beta1 CRDs (incl. extensions) installed?")
            print("    (infra/04-install-agent-sandbox.sh; the fleet needs SandboxTemplate/WarmPool/Claim)")
        elif "forbidden" in msg or "403" in msg:
            print("  → RBAC: this identity can't manage templates/pools or read CRDs/runtimeclasses.")
            print("    Re-run infra/05-setup-rbac.sh (mini_swe_agent_2 needs the extended rules).")
        elif "did not become ready" in msg:
            print(f"  → Pool never Ready. Inspect: kubectl -n {args.namespace} get sandboxwarmpools,sandboxes,pods")
            print("  → If you passed --gvisor, ensure the gVisor node pool exists (infra step 02).")
        return 1
    check("warm pool ready", True)

    handle = None
    try:
        print("\n== Phase 2: claim + basic execute() ==")
        handle = await fleet.acquire(task)
        check("sandbox claimed", handle.pod_name is not None, f"pod={handle.pod_name} host={handle.hostname}")
        env = make_env(fleet, handle, env_cfg)
        r = env.execute("echo hello-from-sandbox")
        check("stdout captured", "hello-from-sandbox" in r["output"], r["output"].strip())
        check("returncode 0 on success", r["returncode"] == 0, f"rc={r['returncode']}")
        check("contract shape {output,returncode}", {"output", "returncode"} <= set(r), str(sorted(r)))

        print("\n== Phase 3: exit-code fidelity (the RL-reward-critical bit) ==")
        check("true -> 0", env.execute("true")["returncode"] == 0)
        check("false -> 1", env.execute("false")["returncode"] == 1)
        check("exit 7 -> 7", env.execute("exit 7")["returncode"] == 7)
        rr = env.execute("echo oops >&2; exit 3")
        check(
            "stderr merged + rc 3",
            "oops" in rr["output"] and rr["returncode"] == 3,
            f"rc={rr['returncode']} out={rr['output'].strip()!r}",
        )

        print("\n== Phase 4: exec prelude (cwd + env exports; the template has no workingDir/env) ==")
        check("cwd is the configured dir", env.execute("pwd")["output"].strip() == args.cwd)
        check("per-call cwd override", env.execute("pwd", cwd="/")["output"].strip() == "/")
        check("config env exported", "hello-env" in env.execute("echo $SMOKE_VAR")["output"])
        env.execute("echo marker > smoke_marker.txt")
        check(
            "filesystem persists across execs (agent-loop pattern)",
            env.execute("cat smoke_marker.txt")["returncode"] == 0,
        )

        print("\n== Phase 5: timeout maps to rc=-1 (never a false success) ==")
        t = env.execute("sleep 5", timeout=2)
        check("timeout -> rc -1 + marker", t["returncode"] == -1 and "timed out" in t["output"], f"rc={t['returncode']}")

        print("\n== Phase 6: eval pattern — release, claim a FRESH box from the same pool ==")
        if args.keep:
            print("  (skipped: --keep)")
        else:
            old_pod = handle.pod_name
            await fleet.release(handle)
            handle = await fleet.acquire(task)
            env2 = make_env(fleet, handle, env_cfg)
            check("second claim succeeded", handle.pod_name is not None, f"pod={handle.pod_name} (was {old_pod})")
            check(
                "eval box is FRESH (marker from box 1 absent)",
                env2.execute("cat smoke_marker.txt")["returncode"] != 0,
            )
    finally:
        if args.keep:
            print(f"\n--keep: leaving pool + claim up. Inspect/clean up:")
            print(f"  kubectl -n {args.namespace} get sandboxtemplates,sandboxwarmpools,sandboxclaims,sandboxes,pods")
            print(f"  # clean up: delete the claim, then re-run without --keep, or:")
            print(f"  kubectl -n {args.namespace} delete sandboxclaims,sandboxwarmpools,sandboxtemplates -l app=agent-sandbox-rl")
        else:
            print("\n== Phase 7: teardown sweeps everything the fleet created ==")
            await fleet.teardown()
            cluster = next(iter(fleet.registry))
            sel = cluster.resources.managed_selector()
            leftovers = (
                cluster.resources.list_claims(label_selector=sel)
                + cluster.resources.list_warmpools(label_selector=sel)
                + cluster.resources.list_templates(label_selector=sel)
            )
            check("no leftover claims/pools/templates", not leftovers, str(leftovers) if leftovers else "")

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n== {'ALL CHECKS PASSED' if not failed else 'FAILURES: ' + ', '.join(failed)} "
          f"({sum(ok for _, ok, _ in _results)}/{len(_results)}) ==")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--namespace", default="skyrl-sandboxes")
    ap.add_argument("--cwd", default="/tmp")
    ap.add_argument("--gvisor", action="store_true")
    ap.add_argument("--keep", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
