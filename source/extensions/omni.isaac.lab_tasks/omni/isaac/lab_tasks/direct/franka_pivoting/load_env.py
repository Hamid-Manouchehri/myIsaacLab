# visualize_franka_cabinet.py
# Just spawn the Franka + cabinet direct env and step with zero actions.

import argparse
from omni.isaac.lab.app import AppLauncher

# ---------- App launcher / CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- Imports AFTER app starts ----------
import torch
from omni.isaac.lab_tasks.direct.franka_pivoting.franka_pivoting_env_custom import (
    FrankaPivotingEnvCfg,
    FrankaPivotingEnv,
)


def main():
    # 1) Build config and override num_envs if you want
    cfg = FrankaPivotingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    # 2) Create env – this sets up SimulationContext, scene, robot, cabinet, etc.
    env = FrankaPivotingEnv(cfg=cfg, render_mode=None)

    # 3) Reset once
    obs, _ = env.reset()

    # Zero actions: shape [num_envs, action_dim]
    actions = torch.zeros((env.num_envs, cfg.action_space), device=env.device)

    print("[INFO] Scene ready. Just visualizing...")

    step_count = 0
    while simulation_app.is_running():
        actions = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
        obs, reward, terminated, truncated, info = env.step(actions)

        # optional auto-reset so it doesn’t hang when episodes end:
        if (terminated | truncated).any():
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
