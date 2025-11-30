# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def cube_height(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene.rigid_objects[cube_cfg.name]
    pos = cube.data.root_pos_w[:, 2:3]  # (num_envs, 1) z height

    # log for RL library
    env.extras.setdefault("log", {})
    env.extras["log"]["cube_height"] = pos  # per-env tensor

    return pos


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_link_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b
