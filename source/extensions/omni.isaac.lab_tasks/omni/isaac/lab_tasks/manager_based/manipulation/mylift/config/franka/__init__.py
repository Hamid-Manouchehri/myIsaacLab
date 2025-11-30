# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="my_Isaac_Lift_Cube_Franka_v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="my_Isaac_Lift_Cube_Franka_Play_v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="my_Isaac_Lift_Cube_Franka_IK_Abs_v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="my_Isaac_Lift_Teddy_Bear_Franka_IK_Abs_v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="my_Isaac_Lift_Cube_Franka_IK_Rel_v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaCubeLiftEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
