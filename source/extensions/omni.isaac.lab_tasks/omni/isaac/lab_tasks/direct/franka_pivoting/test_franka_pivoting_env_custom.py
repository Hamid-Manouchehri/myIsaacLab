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
    observation_space = 23  # dim of observation
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
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

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
        # print("\n\n\n\n\n\n")
        # print(f"[INFO] FrankaPivotingEnv dt: {self.dt}")

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        # print(f"[INFO] robot_dof_lower_limits: {self.robot_dof_lower_limits}")
        # print(f"[INFO] robot_dof_upper_limits: {self.robot_dof_upper_limits}")

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
        # print(f"robot_dof_speed_scales: {self.robot_dof_speed_scales}")

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()  # Get the whole USD stage

        # compute hand_pose in env (local) coordinates
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        # print(f"[INFO] hand_pose: {hand_pose}")

        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        # print(f"[INFO] lfinger_pose: {lfinger_pose}")

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
        # print(f"[INFO] hand_pose_inv_pos: {hand_pose_inv_pos}, hand_pose_inv_rot: {hand_pose_inv_rot}")

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        # print(f"[INFO] robot_local_grasp_pose_rot: {robot_local_grasp_pose_rot}, robot_local_pose_pos: {robot_local_pose_pos}")

        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        # self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        # self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))
        # print(f"[INFO] drawer_local_grasp_pos: {self.drawer_local_grasp_pos}")
        # print(f"[INFO] drawer_local_grasp_rot: {self.drawer_local_grasp_rot}")

        cube_local_grasp_pose = torch.tensor([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],  # 3cm above cube center, adjust to your USD size
            device=self.device)
        self.cube_local_grasp_pos = cube_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.cube_local_grasp_rot = cube_local_grasp_pose[3:7].repeat((self.num_envs, 1))
 
          

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
        # print(f"[INFO] gripper_forward_axis: {self.gripper_forward_axis}")
        # print(f"[INFO] drawer_inward_axis: {self.drawer_inward_axis}")
        # print(f"[INFO] gripper_up_axis: {self.gripper_up_axis}")
        # print(f"[INFO] drawer_up_axis: {self.drawer_up_axis}")

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        # self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]
        # print(f"[INFO] hand_link_idx: {self.hand_link_idx}", f"left_finger_link_idx: {self.left_finger_link_idx}", 
        #       f"right_finger_link_idx: {self.right_finger_link_idx}", f"drawer_link_idx: {self.drawer_link_idx}")

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


    def _setup_scene(self):  
        # Instantiating the assets (robot and cabinet) as Articulation objects from their configs. 
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
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # print(f"[DEBUG] actions: {self.actions}")  # repeats
        # print(f"[DEBUG] robot_dof_targets: {self.robot_dof_targets}")
        # print(f"[DEBUG] targets: {targets}")

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)
        pass

    # post-physics step calls
    """
    Episode termination conditions.
    """
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_pos = self._cube.data.root_pos_w
        cube_quat = self._cube.data.root_quat_w

        # success condition: lifted above 0.95 m AND tilted at least 45 deg
        cube_height = cube_pos[:, 2]
        world_up = torch.tensor([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        cube_up = tf_vector(cube_quat, self.object_up_axis)
        cos_tilt = (cube_up * world_up).sum(dim=-1)
        # cos 45deg ≈ 0.707
        tilted_enough = cos_tilt < 0.7

        terminated = (cube_height > 0.95) & tilted_enough
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_link_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_link_pos_w[:, self.right_finger_link_idx]
        cube_pos = self._cube.data.root_pos_w
        cube_quat = self._cube.data.root_quat_w

        return self._compute_rewards(
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
            self.cfg.open_reward_scale,       # reuse as lift/pivot scale
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
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # print(f"[DEBUG] Reset robot joint_pos: {joint_pos}")
        # print(f"[DEBUG] Reset robot joint_vel: {joint_vel}")  # repeats

        # cube state: randomize position/orientation slightly around nominal
        cube_state = self._cube.data.default_root_state[env_ids].clone()

        # e.g., randomize xy in a small square and small orientation noise
        noise_xy = sample_uniform(-0.02, 0.02, (len(env_ids), 2), self.device)
        # Convert from local to world coordinates by adding env origins
        cube_state[:, 0:3] = cube_state[:, 0:3] + self.scene.env_origins[env_ids]
        cube_state[:, 0:2] += noise_xy  # x, y
        # small yaw noise: you can convert to quaternion or leave zero for now

        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)


        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    """
    What the policy observes.
    """
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.cube_grasp_pos - self.robot_grasp_pos

        cube_pos = self._cube.data.root_pos_w
        cube_quat = self._cube.data.root_quat_w

        # height feature
        cube_height = cube_pos[:, 2].unsqueeze(-1)  # z in world

        # tilt feature: dot(cube_up, world_up), 1=upright, 0=sideways
        world_up = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        cube_up = tf_vector(cube_quat, self.object_up_axis)  # same axis as in reward
        cube_tilt = (cube_up * world_up).sum(dim=-1, keepdim=True)

        obs = torch.cat(
            (
                dof_pos_scaled,                                   # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # 9
                to_target,                                       # 3
                cube_height,                                     # 1
                cube_tilt,                                       # 1
            ),
            dim=-1,
        )

        # print(f"[DEBUG] dof_pos_scaled: {dof_pos_scaled}")  # repeats
        # print(f"[DEBUG] to_target: {to_target}")
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        cube_pos = self._cube.data.root_pos_w[env_ids]
        cube_rot = self._cube.data.root_quat_w[env_ids]
        (self.robot_grasp_rot[env_ids], 
         self.robot_grasp_pos[env_ids],
        self.cube_grasp_rot[env_ids], 
        self.cube_grasp_pos[env_ids]) = self._compute_grasp_transforms(
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
        lift_reward_scale,       # formerly open_reward_scale
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # 1) distance to cube grasp frame
        d = torch.norm(franka_grasp_pos - cube_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward = dist_reward**2
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        # 2) orientation: align gripper with cube side
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(cube_grasp_rot, object_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(cube_grasp_rot, object_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # 3) lift + pivot: encourage height and side tilt
        cube_height = cube_pos[:, 2]
        world_up = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((num_envs, 1))
        cube_up = tf_vector(cube_quat, object_up_axis)
        cos_tilt = (cube_up * world_up).sum(dim=-1)  # 1 upright, 0 sideways

        # lift reward: positive when cube above threshold
        target_height = 0.95  # adjust to your scene
        lift_reward = torch.clamp(cube_height - target_height, min=0.0)

        # pivot reward: maximize 1 - |cos_tilt| to push cube towards 90deg tilt
        pivot_reward = 1.0 - torch.abs(cos_tilt)

        # 4) action penalty
        action_penalty = torch.sum(actions**2, dim=-1)

        # 5) optional: finger “wrap” or closeness along one axis of cube (simple version: drop it)
        finger_dist_penalty = torch.zeros_like(d)  # start without extra hand-shape shaping

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + lift_reward_scale * (lift_reward + pivot_reward)
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "lift_pivot_reward": (lift_reward_scale * (lift_reward + pivot_reward)).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        }

        return rewards


    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        # print(f"[DEBUG] global_franka_pos: {global_franka_pos}", f"global_franka_rot: {global_franka_rot}")  # repeats
        # print(f"[DEBUG] global_drawer_pos: {global_drawer_pos}", f"global_drawer_rot: {global_drawer_rot}")
        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
