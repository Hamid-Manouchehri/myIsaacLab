# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR5E_CFG`: The UR5e arm without a gripper.
* :obj:`UR5E_WITH_ROBOTIQ_2F85_CFG`: The UR5e arm with Robotiq 2F-85 gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##


UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=60.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-5e arm using implicit actuator models."""


# Path to the UR5e with Robotiq 2F-85 gripper USD file
_UR5E_ROBOTIQ_USD_PATH = os.path.normpath(
    "C:/Users/hmanouch/projects/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/ur5e_pivoting/config/ur5e/ur5e_with_gripper.usd"
)

UR5E_WITH_ROBOTIQ_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.normpath(
    "C:/Users/hmanouch/projects/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/ur5e_pivoting/config/ur5e/ur5e_with_gripper.usd"
),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            # Robotiq 2F-85 gripper joints (open position: ~0.085m, closed: 0.0m)
            # Adjust joint names based on your USD file
            "finger_1_joint": 0.085,
            "finger_2_joint": 0.085,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=60.0,
            stiffness=0.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger.*joint", ".*finger.*"],
            velocity_limit=100.0,
            effort_limit=235.0,  # Max force for Robotiq 2F-85
            stiffness=1e3,
            damping=50.0,
        ),
    },
)
"""Configuration of UR-5e arm with Robotiq 2F-85 gripper using implicit actuator models.

Note: Joint names may need adjustment based on your USD file.
Common joint name variations for Robotiq 2F-85:
- finger_1_joint, finger_2_joint
- finger_joint_1, finger_joint_2
- left_finger_joint, right_finger_joint
- robotiq_finger_1_joint, robotiq_finger_2_joint
"""
