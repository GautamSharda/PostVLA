from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class JaxOpenPiPolicy(torch.nn.Module, BasePolicy):
    """Eval-only OpenPI JAX policy adapter for RLinf embodied rollout workers."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config

        openpi_cfg = _cfg_get(cfg, "openpi", None)
        config_name = _cfg_get(openpi_cfg, "config_name", "pi05_so100_sim")
        checkpoint_dir = download.maybe_download(str(cfg.model_path))

        train_config = openpi_config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
        self.action_dim = int(_cfg_get(cfg, "action_dim", 6))
        self.num_action_chunks = int(_cfg_get(cfg, "num_action_chunks", 16))

    def forward(self, *args, **kwargs):
        return self.default_forward(*args, **kwargs)

    def default_forward(self, **kwargs):
        raise RuntimeError(
            "jax_openpi is eval/reference only. PPO training requires a trainable "
            "PyTorch policy that can recompute logprobs and values."
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if mode != "eval":
            raise RuntimeError(
                "jax_openpi only supports eval rollout. Do not use it as the PPO "
                "actor; it cannot provide trainable logprobs or values."
            )

        states = _to_numpy(env_obs["states"])
        main_images = _to_numpy(env_obs["main_images"])
        wrist_images = _to_numpy(env_obs["wrist_images"])
        prompts = env_obs["task_descriptions"]

        batch_actions = []
        for i in range(states.shape[0]):
            obs = {
                "observation/exterior_image_1_left": main_images[i],
                "observation/wrist_image_left": wrist_images[i],
                "observation/joint_position": states[i, :5].astype(np.float32),
                "observation/gripper_position": states[i, 5:6].astype(np.float32),
                "prompt": prompts[i],
            }
            actions = np.asarray(self.policy.infer(obs)["actions"])
            if actions.ndim == 3:
                actions = actions[0]
            batch_actions.append(
                actions[: self.num_action_chunks, : self.action_dim]
            )

        return np.stack(batch_actions, axis=0), {}


def get_model(cfg: DictConfig, torch_dtype=None) -> JaxOpenPiPolicy:
    del torch_dtype
    return JaxOpenPiPolicy(cfg)
