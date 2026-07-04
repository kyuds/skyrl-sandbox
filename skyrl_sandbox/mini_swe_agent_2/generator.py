"""Gen-2 mini-swe-agent generator: same SkyRL generator interface, agent-sandbox-rl rollout backend.

Differences from :mod:`skyrl_sandbox.mini_swe_agent.generator` (gen-1), by design:

* **No per-trajectory Ray task.** Gen-1 fans out ``init_and_run.remote`` and each task cold-creates
  its own ``Sandbox`` CR. A warm-pool fleet is stateful and unserializable (kube clients, claim
  bookkeeping), so it cannot be closed over by Ray tasks; instead ONE ``AsyncSandboxFleet`` lives in
  this generator and ``fleet.run(process_fn, strategy)`` drives the whole batch — per-image warm
  pools sized to ``max_concurrent``, claims per trajectory, bounded parallelism, per-task error
  capture, and a RunReport, all from agent-sandbox-rl. Rollouts are I/O-bound (litellm HTTP +
  pod-exec), so the asyncio loop + a sized thread pool replaces the Ray fan-out.
* **Two claims per trajectory** (rollout box, then a fresh eval box) from the same warm pool, with
  the rollout box released *before* the eval claim — gen-1 held both simultaneously.

Token/loss-mask handling, trajectory saving, and the litellm model contract are copied from gen-1
verbatim; the interface (``generate(GeneratorInput) -> GeneratorOutput``) is identical.
"""

import asyncio
import copy
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj

from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl.train.generators.base import BatchMetadata
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)

from .environment import Fleet2EnvironmentConfig
from .fleet_util import build_fleet, build_task
from .utils import evaluate_trajectory_on_fleet, make_env, run_env_startup_command

import logging

logger = logging.getLogger(__name__)


@dataclass
class MiniSWE2GeneratorConfig(GeneratorConfig):
    """Extended generator config with Mini-SWE-Agent-specific fields (same knobs as gen-1; the
    sandbox/fleet knobs live in the task YAML's ``environment:`` block, see
    :class:`~skyrl_sandbox.mini_swe_agent_2.environment.Fleet2EnvironmentConfig`)."""

    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""
    miniswe_litellm_model_name: str = ""
    """Optional: the **full, provider-qualified litellm model id** sent to litellm, used verbatim and
    DECOUPLED from ``trainer.policy.model.path`` (which must stay a valid HF id because it loads the
    tokenizer). E.g. ``fireworks_ai/accounts/fireworks/models/...`` (auth via ``FIREWORKS_AI_API_KEY``).
    Empty = ``openai/<model.path>`` (the SkyRL local-vLLM default via ``OPENAI_BASE_URL`` — the
    training path). See gen-1's field docstring for the tokenizer stand-in caveat."""


class DefaultAgentWithReminder(DefaultAgent):
    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the output."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."

        self.add_message("user", observation)
        return output


class MiniSweAgent2Generator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)

        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # Full provider-qualified litellm model id (used verbatim), decoupled from model.path (the
        # tokenizer). Empty -> openai/<model.path> (SkyRL local-vLLM default).
        self.litellm_model_name = generator_cfg.miniswe_litellm_model_name or ("openai/" + self.model_name)

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSweAgent2Generator doesn't support custom chat template")

        # Built lazily on first generate(): the fleet knobs live in the task YAML's environment
        # block, and the fleet itself is reusable across batches (teardown resets its plan/pools).
        self._env_cfg: Optional[Fleet2EnvironmentConfig] = None
        self._fleet = None
        self._executor: Optional[ThreadPoolExecutor] = None

    # --- fleet plumbing ------------------------------------------------------------------------

    def _ensure_fleet(self, sweagent_config: dict) -> None:
        if self._fleet is not None:
            return
        self._env_cfg = Fleet2EnvironmentConfig.from_dict(sweagent_config.get("environment", {}), logger)
        self._fleet = build_fleet(self._env_cfg)
        # asyncio.to_thread runs on the loop's default executor (~32 threads by default). Each
        # in-flight rollout parks one thread on blocking I/O (litellm HTTP / pod-exec websocket) and
        # the fleet threads its own k8s calls, so size the pool to the concurrency budget.
        self._executor = ThreadPoolExecutor(
            max_workers=max(32, 2 * self._env_cfg.max_concurrent + 8),
            thread_name_prefix="miniswe2-rollout",
        )

    async def _run_one_trajectory(
        self,
        task,
        handle,
        instance: dict,
        data_source: str,
        sweagent_config: dict,
        sampling_params: Dict[str, Any],
        repetition_id: int,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[dict], float, Optional[str]]:
        """One rollout on an already-claimed warm sandbox: agent loop -> fresh-box eval -> save.

        Mirrors gen-1's ``init_and_run`` body. Runs on the shared event loop; every blocking step
        (agent loop, execs, file writes) is pushed to a worker thread.
        """
        # Per-trajectory copy: gen-1 got isolation for free from Ray task serialization; here the
        # batch shares one parsed YAML dict and litellm sampling params are injected per trajectory.
        model_config = copy.deepcopy(sweagent_config.get("model", {}))
        model_config.setdefault("model_kwargs", {}).update(sampling_params)
        model = get_model(self.litellm_model_name, model_config)

        agent = None
        extra_info = None
        result = None
        reward = 0
        error = None
        exit_status = None  # pre-bound: the finally block saves it even if a BaseException escapes
        try:
            env = make_env(self._fleet, handle, self._env_cfg)
            await asyncio.to_thread(run_env_startup_command, env, sweagent_config, instance)
            agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
            exit_status, result = await asyncio.to_thread(agent.run, instance["problem_statement"])
        except Exception as e:
            logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
            exit_status, result = type(e).__name__, str(e)
            error = str(e)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            # The rollout box is done (the submission diff is in `result`): hand it back BEFORE
            # claiming the eval box, so peak demand stays ~max_concurrent (gen-1 overlapped the two).
            # The managed runner's own release of this handle then becomes a documented no-op.
            try:
                await self._fleet.release(handle)
            except Exception:
                logger.warning("failed to release rollout sandbox for %s", instance.get("instance_id"), exc_info=True)

            path = Path(self.generator_cfg.miniswe_traj_dir) / f"step_{batch_metadata.global_step}" / batch_metadata.training_phase
            filename = f"{instance['instance_id']}_{repetition_id}.json"
            if agent is not None:
                eval_error = None
                try:
                    result = await evaluate_trajectory_on_fleet(
                        self._fleet, task, instance, result, sweagent_config, self._env_cfg
                    )
                    reward = int(result["resolved"])
                    eval_error = result["eval_error"]
                    if eval_error:
                        error = eval_error
                        logger.debug(f"Error during evaluation {eval_error}")
                except Exception as e:
                    logger.debug(f"Error during evaluation {e}")
                    logger.debug(f"traceback: {traceback.format_exc()}")
                    eval_error = str(e)
                    error = str(e)

                def _save() -> None:
                    path.mkdir(parents=True, exist_ok=True)
                    save_traj(agent, path / filename, exit_status=exit_status, result=result, extra_info=extra_info, reward=reward, eval_error=eval_error)  # type: ignore[arg-type]

                await asyncio.to_thread(_save)

        return (agent.messages if agent is not None else [], reward, error)

    # --- token plumbing (copied from gen-1) ------------------------------------------------------

    def _postprocess_trajectory(
        self,
        messages: List[dict],
        reward: float,
        max_tokens: int,
        max_input_length: int,
    ) -> Tuple[Optional[List[int]], Optional[float], Optional[str], Optional[List[int]], Optional[List[int]], None]:
        """messages -> (response_ids, reward, stop_reason, loss_mask, prompt_ids, None); gen-1 verbatim."""
        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        while response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        prompt_ids = initial_input_ids
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    # --- entry point -----------------------------------------------------------------------------

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch (``fleet.run`` preserves task order and
        captures per-task exceptions as results; a captured exception = a failed trajectory, filtered
        exactly like gen-1's empty-messages case).
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
        )
        # TODO (kyuds): gate this logic behind a flag.
        # SkyRL only knows the 'vllm' backend, so it always emits vLLM-only sampling params. When the LLM
        # is reached via litellm against a hosted OpenAI-compatible API (Fireworks), those are rejected
        # ("Extra inputs are not permitted"), and vLLM sentinels are out of range (top_k=-1 means "no
        # filtering" in vLLM but Fireworks requires 0..100). Sanitize for the litellm path.
        for _k in ("skip_special_tokens", "include_stop_str_in_output", "min_tokens"):
            sampling_params.pop(_k, None)
        _tk = sampling_params.get("top_k")
        if not (isinstance(_tk, int) and 0 <= _tk <= 100):
            sampling_params.pop("top_k", None)  # drop vLLM's -1 sentinel; let the provider default

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        self._ensure_fleet(sweagent_config)
        # to_thread uses the loop's default executor; install the sized one (idempotent, per loop).
        asyncio.get_running_loop().set_default_executor(self._executor)

        # One fleet Task per trajectory; per-trajectory context stays here, keyed by batch index.
        contexts = []
        tasks = []
        for i in range(len(prompts)):
            # NOTE: input `prompts` are unused (gen-1 parity) — mini-swe-agent rebuilds the prompt
            # from the instance's problem_statement + templates.
            instance = env_extras[i]["instance"]
            data_source = env_extras[i]["data_source"]
            tasks.append(build_task(instance, data_source, trajectory_ids[i].repetition_id, i))
            contexts.append((instance, data_source, trajectory_ids[i].repetition_id))

        async def process_fn(task, handle):
            instance, data_source, repetition_id = contexts[task.metadata["index"]]
            return await self._run_one_trajectory(
                task,
                handle,
                instance,
                data_source,
                sweagent_config,
                sampling_params,
                repetition_id,
                batch_metadata,
            )

        self._fleet.load_tasks(tasks)
        # Warm pools per image, claim per trajectory, bounded parallelism, teardown — all inside run().
        rollout_results = await self._fleet.run(process_fn, strategy=self._env_cfg.warmpool_strategy)

        all_outputs = []
        num_failed = 0
        for result in rollout_results:
            # fleet.run captures per-task infra failures (acquire/pool errors) as the result value.
            if result is None or isinstance(result, BaseException) or not len(result[0]):
                num_failed += 1
                all_outputs.append((None, None, None, None, None, None))
                continue
            messages, reward, _error = result
            all_outputs.append(self._postprocess_trajectory(messages, reward, max_tokens, max_input_length))
        if num_failed:
            logger.warning(f"{num_failed}/{len(rollout_results)} trajectories failed and were dropped from the batch")

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[0] is not None]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
