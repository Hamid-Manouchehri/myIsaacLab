# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This file is how to define the RL task.
"""

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform


"""
Cnfiguration of the env (sim, scene, robot, cabinet, rewards)
- episode length
- dt
- num of envs
- robot
- cabinet
- terrain
- reward scale
"""
@configclass
class FrankaPivotingEnvCfg(DirectRLEnvCfg):
    """
    env: Describe what the world looks like and how the RL problem is defined.
    - Holding parameters for simulation, scene layout, robot, cabinet, terrain, rewards, etc.
    """
    episode_length_s = 8.3333  # 500 timesteps, how long an episode is in sim steps
    decimation = 2  # one policy step for every two physics steps
    action_space = 9  # dim of action
    observation_space = 24  # dim of observation: 9 (dof_pos) + 9 (dof_vel) + 3 (to_target) + 1 (cube_height) + 2 (finger_pos) = 24
    state_space = 0

    # simulation parameters
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene layout
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot definition
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(  # initial joint poses
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, -0.2, 0.8),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet definition
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # cube object
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ToyCube",   # global path with regex
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(  # Local pose relative to env origin
            pos=(0., 0.15, 0.9),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


    # ground plane / terrain definition
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 2.0
    rot_reward_scale = 0.5  # reduced since orientation doesn't matter much
    lift_reward_scale = 15.0  # increased to encourage lifting
    grasp_reward_scale = 5.0  # reward for successful grasp
    action_penalty_scale = 0.05
    finger_reward_scale = 3.0  # increased to encourage finger closing

"""
Actual RL loop callback / env:
- actions
- rewards / dones
- observations
- reset
"""
class FrankaPivotingEnv(DirectRLEnv):
    """
    Implement the RL environment logic for one simulation step (How one RL step is computed).
    """
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaPivotingEnvCfg

    def __init__(self, cfg: FrankaPivotingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates from USD prim world pose"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()  # Get the whole USD stage

        # compute hand_pose in env (local) coordinates (env origin in world coordinates: self.scene.env_origins)
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )

        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )

        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )
        # print(f"[INFO] rfinger_pose: {rfinger_pose}")   

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        # print(f"[INFO] finger_pose: {finger_pose}")
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # Orientation and position of the robot's grasp frame relative to the hand frame
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        # print(f"[INFO] robot_local_grasp_pose_rot: {robot_local_grasp_pose_rot}, robot_local_pose_pos: {robot_local_pose_pos}")

        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))  # Create copies per env.
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # Cube grasp pose: approach from the side (y-axis) at a slight offset
        # This allows the gripper to approach and grasp the cube from the side
        cube_local_grasp_pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # at cube center
            device=self.device)
        self.cube_local_grasp_pos = cube_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.cube_local_grasp_rot = cube_local_grasp_pose[3:7].repeat((self.num_envs, 1))
 
          
        # Define direction vectors in local frame for the gripper and the object, used for orientation rewards.
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.object_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.object_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        # We use those indices later to read positions/rotations of those specific links for rewards, observations, etc.
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # Buffers to store the grasp poses in world coordinates per env:
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


    def _setup_scene(self):  
        """
        Instantiating the assets (robot and cabinet) as Articulation objects from their configs. 
        Building the scene once before the RL loop (training).
        """
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self._cube = self.cfg.cube.class_type(self.cfg.cube)

        # Registering the articulations and rigid objects to the scene for simulation.
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet
        self.scene.rigid_objects["cube"] = self._cube

        # Multi-env cloning:
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    """
    How the policy actions map to actuator commands (joint-space or cartesian-space).
    """
    def _pre_physics_step(self, actions: torch.Tensor):
        """Maps the normalized actions from the policy to the robot's joint position targets."""
        self.actions = actions.clone().clamp(-1.0, 1.0)  # create a separate copy of actions and clamps / bounds it within [-1, 1]
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale  # Treat actions as scaled joint velocity commands.
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)


    # post-physics step calls
    """
    Episode termination conditions.
    """
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_pos = self._cube.data.root_pos_w

        # success condition: lifted above 1.0 m
        cube_height = cube_pos[:, 2]
        terminated = cube_height > 1.0

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()  # updates cached grasp poses
        robot_left_finger_pos = self._robot.data.body_link_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_link_pos_w[:, self.right_finger_link_idx]
        cube_pos = self._cube.data.root_pos_w
        cube_quat = self._cube.data.root_quat_w

        return self._compute_rewards(  # Compute scalar rewards per env.
            self.actions,
            cube_pos,
            cube_quat,
            self.robot_grasp_pos,
            self.cube_grasp_pos,
            self.robot_grasp_rot,
            self.cube_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.object_inward_axis,
            self.gripper_up_axis,
            self.object_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.grasp_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )


    """
    Resetting envs:
    - Randomizes robot joint positions near default → exploration.
    - Resets cabinet joints to closed.
    - Writes everything into sim.
    - Updates cached grasp poses.
    """
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            return
        super()._reset_idx(env_ids)  # clear episode-level buffers for each env
        # robot state - add slight randomization for exploration
        num_reset = len(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.1,
            0.1,
            (num_reset, self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)  # zero-initial velocity
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cube state - add slight randomization in position (small xy offset, keep z the same)
        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0:3] = cube_state[:, 0:3] + self.scene.env_origins[env_ids]
        # Add small random offset in x and y (within 5cm)
        cube_state[:, 0:2] += sample_uniform(
            -0.05,
            0.05,
            (num_reset, 2),
            self.device,
        )
        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)
        self._compute_intermediate_values(env_ids)


    def _get_observations(self) -> dict:
        # What the policy observes at each RL step.
        dof_pos_scaled = (  # Normalized joint positions to [-1, 1]
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        to_target = self.cube_grasp_pos - self.robot_grasp_pos  # 3D vector from robot grasp to cube grasp (relative position error)

        cube_pos = self._cube.data.root_pos_w
        # cube_quat = self._cube.data.root_quat_w

        # height feature
        cube_height = cube_pos[:, 2].unsqueeze(-1)  # z in world

        # tilt feature: dot(cube_up, world_up), 1=upright, 0=sideways
        # world_up = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # cube_up = tf_vector(cube_quat, self.object_up_axis)  # same axis as in reward
        # cube_tilt = (cube_up * world_up).sum(dim=-1, keepdim=True)

        # Finger positions (normalized) to help policy learn when to close fingers
        finger_joint1_idx = self._robot.find_joints("panda_finger_joint1")[0]
        finger_joint2_idx = self._robot.find_joints("panda_finger_joint2")[0]
        finger_pos = self._robot.data.joint_pos[:, [finger_joint1_idx, finger_joint2_idx]]
        # Normalize finger positions (0.0 = fully open, 0.04 = default open, normalize to [-1, 1])
        finger_pos_normalized = (finger_pos / 0.04) * 2.0 - 1.0  # maps [0, 0.04] to [-1, 1]
        finger_pos_normalized = finger_pos_normalized.squeeze()

        obs = torch.cat(
            (
                dof_pos_scaled,                                   # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # 9
                to_target,                                       # 3
                cube_height,                                     # 1
                finger_pos_normalized,                           # 2
            ),
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)  # Use slice instead of protected member

        hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        cube_pos = self._cube.data.root_pos_w[env_ids]
        cube_rot = self._cube.data.root_quat_w[env_ids]
        (self.robot_grasp_rot[env_ids], 
         self.robot_grasp_pos[env_ids],
        self.cube_grasp_rot[env_ids], 
        self.cube_grasp_pos[env_ids]) = self._compute_grasp_transforms(  # Compute global grasp poses in world frame for each env.
            hand_rot, hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            cube_rot, cube_pos,
            self.cube_local_grasp_rot[env_ids],
            self.cube_local_grasp_pos[env_ids],
        )

    """
    Define the reward logic.
    """
    def _compute_rewards(
        self,
        actions,
        cube_pos,
        cube_quat,
        franka_grasp_pos,
        cube_grasp_pos,
        franka_grasp_rot,
        cube_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        object_inward_axis,
        gripper_up_axis,
        object_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        lift_reward_scale,
        grasp_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # 1) Distance to cube: encourage approaching the cube
        d = torch.norm(franka_grasp_pos - cube_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward = dist_reward**2
        dist_reward = torch.where(d <= 0.05, dist_reward * 2, dist_reward)  # bonus for within 5cm

        # 2) Orientation: optional alignment (reduced weight since orientation doesn't matter)
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(cube_grasp_rot, object_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(cube_grasp_rot, object_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # 3) Finger closing reward: encourage closing fingers when near the cube
        # Get finger joint positions (last two joints are fingers)
        finger_joint1_pos = joint_positions[:, -2]  # left finger
        finger_joint2_pos = joint_positions[:, -1]  # right finger
        finger_open_pos = 0.035  # default open position
        
        # Calculate how closed the fingers are (0 = fully open, 1 = fully closed)
        finger1_closed = torch.clamp((finger_open_pos - finger_joint1_pos) / finger_open_pos, 0.0, 1.0)
        finger2_closed = torch.clamp((finger_open_pos - finger_joint2_pos) / finger_open_pos, 0.0, 1.0)
        finger_closed_avg = (finger1_closed + finger2_closed) / 2.0
        
        # Reward closing fingers when near the cube (within 10cm)
        near_cube = d < 0.10
        finger_reward = finger_closed_avg * near_cube.float()

        # 4) Lift reward: progressive reward for lifting the cube
        cube_height = cube_pos[:, 2]
        initial_height = 0.9  # initial cube height from config
        target_height = 1.0  # target lift height
        
        # Progressive reward: any height increase is rewarded, with bonus for reaching target
        height_increase = cube_height - initial_height
        lift_reward = torch.clamp(height_increase / (target_height - initial_height), 0.0, 1.0)
        # Bonus for reaching target height
        lift_reward = torch.where(cube_height >= target_height, lift_reward + 0.5, lift_reward)

        # 5) Grasp reward: reward successful grasp (cube lifted while fingers are closed)
        # A grasp is successful if:
        # - Cube is above initial height (being lifted)
        # - Fingers are closed (both fingers closed more than 50%)
        # - Cube is near gripper (within 5cm)
        fingers_closed = (finger_closed_avg > 0.5) & (d < 0.05)
        cube_lifted = cube_height > initial_height + 0.02  # at least 2cm above initial
        is_grasped = fingers_closed & cube_lifted
        grasp_reward = is_grasped.float()

        # 6) Action penalty: discourage excessive actions
        action_penalty = torch.sum(actions**2, dim=-1)

        # Combine all rewards
        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + lift_reward_scale * lift_reward
            + grasp_reward_scale * grasp_reward
            + finger_reward_scale * finger_reward
            - action_penalty_scale * action_penalty
        )

        # Optional: log rewards for debugging
        # self.extras["log"] = {
        #     "dist_reward": (dist_reward_scale * dist_reward).mean(),
        #     "rot_reward": (rot_reward_scale * rot_reward).mean(),
        #     "lift_reward": (lift_reward_scale * lift_reward).mean(),
        #     "grasp_reward": (grasp_reward_scale * grasp_reward).mean(),
        #     "finger_reward": (finger_reward_scale * finger_reward).mean(),
        #     "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        # }

        return rewards


    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        cube_rot,
        cube_pos,
        cube_local_grasp_rot,
        cube_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            cube_rot, cube_pos, cube_local_grasp_rot, cube_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
