"""Generate-only entrypoint for the gen-2 (agent-sandbox-rl) example (rollouts, NO training).

Runs ONLY the SkyRL generator against a LiteLLM-compatible endpoint, so you can validate the
mini-swe-agent + agent-sandbox-rl warm-pool path end-to-end **without GPUs or weight updates**.
Mirror of :mod:`skyrl_sandbox.mini_swe_agent.generate` with the gen-2 generator.
"""

import asyncio
import sys

import ray
from skyrl.train.config import SkyRLGymConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.evaluate import evaluate, evaluate_step_wise
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import validate_generator_cfg
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInterface

from .generator import MiniSweAgent2Generator, MiniSWE2GeneratorConfig

MiniSWE2Config = make_config(generator_cls=MiniSWE2GeneratorConfig)


class MiniSWE2GenerateExp(BasePPOExp):
    """Eval/generate-only: build just the generator + inference client, run ``evaluate``, no trainer."""

    def get_train_dataset(self):
        # No training data: this avoids BasePPOExp requiring data.train_data for a generate-only run.
        return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return MiniSweAgent2Generator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )

    async def run(self, inference_engine_client: InferenceEngineInterface) -> dict:
        assert self.eval_dataset is not None, "generate-only requires an eval dataset (set data.val_data)"
        # The generator drives the LLM via litellm directly (Fireworks / remote OpenAI), not the
        # SkyRL inference engine, so this path runs with no engines (num_engines=0). Only wake real
        # engines (wake_up() HTTP-POSTs each remote engine and asserts >0 engines, so skip it here).
        if getattr(inference_engine_client, "engines", None):
            await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        eval_fn = evaluate_step_wise if self.cfg.generator.step_wise_trajectories else evaluate
        results = await eval_fn(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )
        self.get_tracker().log(results, step=0, commit=True)
        return results


@ray.remote(num_cpus=1)
def generate_entrypoint(cfg) -> dict:
    exp = MiniSWE2GenerateExp(cfg)
    # Build the inference client from a sync context (mirrors main_generate.py's eval_entrypoint).
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = MiniSWE2Config.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(generate_entrypoint.remote(cfg))
    print(f"[mini-swe-2 generate-only] metrics: {metrics}")


if __name__ == "__main__":
    main()
