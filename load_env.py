from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

# start app
app = AppLauncher().app

env_id = "Isaac-Lift-Cube-Franka-v0"  # TODO


def main():
    env_cfg = parse_env_cfg(env_id,
                            device="cuda:0", num_envs=1)
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="human")

    obs, _ = env.reset()
    print("Obs keys:", obs.keys())
    for _ in range(10):
        import torch
        action = torch.zeros_like(env.action_manager.action)
        obs, rew, terminated, truncated, info = env.step(action)
        print("reward[0] =", rew[0].item())

    env.close()

if __name__ == "__main__":
    main()
    app.close()
