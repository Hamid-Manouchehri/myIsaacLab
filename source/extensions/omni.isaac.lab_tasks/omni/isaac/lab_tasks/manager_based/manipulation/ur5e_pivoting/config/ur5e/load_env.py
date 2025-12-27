from __future__ import annotations

import torch

from omni.isaac.lab.app import AppLauncher


def main():
    # IMPORTANT: all Omni/Isaac imports happen AFTER AppLauncher is created
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab_tasks.manager_based.manipulation.ur5e_pivoting.config.ur5e.joint_pos_env_cfg import (
        UR5ePivotingEnvCfg,
    )

    # 1) Build env cfg (your manager-based Franka lift config)
    cfg = UR5ePivotingEnvCfg()
    env_cfg = parse_env_cfg(cfg)

    # 2) Create the ManagerBasedRLEnv directly (no gym.make)
    env = ManagerBasedRLEnv(env_cfg)

    # 3) Run a short rollout with zero actions
    obs, info = env.reset()
    print("Env reset. Observation shape:", obs["policy"].shape if isinstance(obs, dict) else obs.shape)

    num_steps = 500
    for step in range(num_steps):
        # zero actions (shape: [num_envs, action_dim])
        actions = torch.zeros(
            (env.num_envs, env.action_space.shape[-1]),
            device=env.device,
            dtype=torch.float32,
        )
        obs, reward, terminated, truncated, info = env.step(actions)

        # reset done envs
        done = torch.logical_or(terminated, truncated)
        if done.any():
            env.reset(done)

    env.close()
    print("Done.")


if __name__ == "__main__":
    # Start Kit/Isaac app first so omni.kit / omni.isaac.* exist
    app_launcher = AppLauncher(headless=False)
    app = app_launcher.app

    try:
        main()
    finally:
        app.close()
