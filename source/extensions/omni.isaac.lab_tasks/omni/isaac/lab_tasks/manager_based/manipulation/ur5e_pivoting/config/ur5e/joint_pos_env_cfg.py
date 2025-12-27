# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.ur5e_pivoting.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.ur5e_pivoting.pivoting_env_cfg import PivotingEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import UR5E_CFG, UR5E_WITH_ROBOTIQ_2F85_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR5ePivotingEnvCfg(PivotingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tool0"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "tool0"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR5ePivotingWithRobotiqEnvCfg(PivotingEnvCfg):
    """UR5e with Robotiq 2F-85 gripper configuration."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur5e with Robotiq gripper
        self.scene.robot = UR5E_WITH_ROBOTIQ_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tool0"]
        # override actions - arm joints only (exclude gripper joints)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            scale=0.5,
            use_default_offset=True,
        )
        # add gripper action
        # Adjust joint names based on your USD file (common variations listed in the config)
        # Common joint name patterns: finger_1_joint, finger_2_joint, finger_joint_1, etc.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger.*joint", ".*finger.*"],  # Regex patterns to match gripper joints
            open_command_expr={"finger.*joint": 0.085, ".*finger.*": 0.085},  # Open: 85mm (max opening)
            close_command_expr={"finger.*joint": 0.0, ".*finger.*": 0.0},  # Closed: 0mm
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "tool0"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR5ePivotingEnvCfg_PLAY(UR5ePivotingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
