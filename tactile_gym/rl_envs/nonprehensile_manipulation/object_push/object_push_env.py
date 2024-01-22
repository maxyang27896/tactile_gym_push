from abc import abstractmethod
from distutils.log import warn
import os, sys
import re
import random

import gym
import numpy as np

from gym import Env, GoalEnv

from tactile_gym.assets import get_assets_path, add_assets_path
from tactile_gym.utils.general_utils import get_orn_diff, quaternion_multiply
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.rest_poses import (
    rest_poses_dict,
)
from tactile_gym.rl_envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv

# Default env modes optimised of MBRL
env_modes_default={
    'movement_mode':'TyRz',
    'control_mode':'TCP_position_control',
    'x_speed_ratio': 1.0,
    'y_speed_ratio': 1.0,
    'Rz_speed_ratio': 1.0,
    'rand_init_orn': False,
    'rand_init_pos_y': False,
    'rand_obj_mass': False,
    'observation_mode':'tactile_pose_data',
    'reward_mode':'dense',
    'importance_obj_goal_pos': 1.0,
    'importance_obj_goal_orn': 1.0,
    'importance_tip_obj_orn': 1.0,
    "terminate_early": True,
    'terminate_terminate_early': True,
    'terminated_early_penalty': -100,
    'reached_goal_reward': 100,
    'max_no_contact_steps': 40, 
    'max_tcp_to_obj_orn': 30/180 * np.pi,
    'mpc_goal_orn_update': False,
    'eval_mode': False,
    'eval_num': 0,
    'goal_list': [],
}

class ObjectPushEnv(BaseObjectEnv, GoalEnv):
    def __init__(
        self,
        max_steps=1000,
        image_size=[64, 64],
        env_modes=env_modes_default,
        obs_stacked_len=1,
        show_gui=False,
        show_tactile=False,
        **kwargs,
    ):

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(
            np.floor(self._control_rate / self._sim_time_step)
        )
        self._max_blocking_pos_move_steps = 20

        # pull params from env_modes specific to push env
        self.rand_init_orn = env_modes["rand_init_orn"]
        self.rand_init_pos_y = env_modes["rand_init_pos_y"]
        self.rand_obj_mass = env_modes["rand_obj_mass"]

        self.terminate_early = env_modes["terminate_early"] if "terminate_early" in env_modes else env_modes_default["terminate_early"]
        self.terminate_terminate_early = env_modes["terminate_terminate_early"] if "terminate_terminate_early" in env_modes else env_modes_default["terminate_terminate_early"]

        self.mpc_goal_orn_update = env_modes["mpc_goal_orn_update"] if "mpc_goal_orn_update" in env_modes else env_modes_default["mpc_goal_orn_update"]

        self.eval_mode = env_modes["eval_mode"] if "eval_mode" in env_modes else env_modes_default["eval_mode"]
        self.eval_num = env_modes["eval_num"] if "eval_num" in env_modes else env_modes_default["eval_num"]
        self.goal_list = env_modes["goal_list"] if "goal_list" in env_modes else env_modes_default["goal_list"]

        # Set speed
        self.x_speed_ratio = env_modes['x_speed_ratio']  if 'x_speed_ratio' in env_modes else env_modes_default['x_speed_ratio']
        self.y_speed_ratio = env_modes['y_speed_ratio']  if 'y_speed_ratio' in env_modes else env_modes_default['y_speed_ratio']
        self.Rz_speed_ratio = env_modes['Rz_speed_ratio']  if 'Rz_speed_ratio' in env_modes else env_modes_default['Rz_speed_ratio']

        # set which robot arm to use
        self.arm_type = "ur5"

        # which tactip to use
        self.tactip_type = "right_angle"
        self.tactip_core = "fixed"
        self.tactip_dynamics = {'stiffness': 50, 'damping': 100, 'friction':10.0}

        # Termination condition parameters
        self.termination_pos_dist = 0.025
        self.terminated_early_penalty = env_modes['terminated_early_penalty']  if 'terminated_early_penalty' in env_modes else env_modes_default['terminated_early_penalty']
        self.max_no_contact_steps = env_modes['max_no_contact_steps']  if 'max_no_contact_steps' in env_modes else env_modes_default['max_no_contact_steps']
        self.reached_goal_reward = env_modes['reached_goal_reward']  if 'reached_goal_reward' in env_modes else env_modes_default['reached_goal_reward']
        self.max_tcp_to_obj_orn =  env_modes['max_tcp_to_obj_orn']  if 'max_tcp_to_obj_orn' in env_modes else env_modes_default['max_tcp_to_obj_orn']

        # Reward parameters
        self.importance_obj_goal_pos = env_modes['importance_obj_goal_pos']  if 'importance_obj_goal_pos' in env_modes else env_modes_default['importance_obj_goal_pos']
        self.importance_obj_goal_orn = env_modes['importance_obj_goal_orn']  if 'importance_obj_goal_orn' in env_modes else env_modes_default['importance_obj_goal_orn']
        self.importance_tip_obj_orn = env_modes['importance_tip_obj_orn']  if 'importance_tip_obj_orn' in env_modes else env_modes_default['importance_tip_obj_orn']

        # Define max values for normalising rewards
        self.max_obj_dist_goal_pos = 0.3
        self.max_obj_dist_goal_orn = np.deg2rad(180)
        self.max_obj_dist_tip_orn = np.deg2rad(90)

        # turn on goal visualisation
        self.visualise_goal = False
        self.current_goal_reached = False
        self.single_goal_reached = False

        # work frame origin
        self.workframe_pos = np.array([0.55, -0.15, 0.04])
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # limits
        TCP_lims = np.array(env_modes['tcp_lims'])
        self.goal_edges = env_modes['goal_edges']
        goal_ranges = env_modes['goal_ranges']
        self.goal_x_min = goal_ranges[0]
        self.goal_x_max = goal_ranges[1]
        self.goal_y_min = goal_ranges[2]
        self.goal_y_max = goal_ranges[3]
        self.goal_sample_eps = 0.3

        if self.eval_mode:
            if self.goal_list:
                self.eval_goals = self.goal_list
            else:
                self.eval_goals = self.make_n_point_goals(self.eval_num)
            self.eval_goal_index = 0

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.tactip_type]["trajectory"]
        
        super(ObjectPushEnv, self).__init__(
            max_steps,
            image_size,
            env_modes,
            obs_stacked_len,
            TCP_lims,
            rest_poses,
            show_gui,
            show_tactile,
        )
        # lod all the objects for trajectory
        self.load_trajectory()

        # this is needed to set some variables used for initial observation/obs_dim()
        self.reset()

        # set the observation space dependent on
        self.setup_observation_space()

        # Draw TCP limits into the environemnt
        self.robot.arm.draw_TCP_box()

    def setup_action_space(self):
        """
        Sets variables used for making network predictions and
        sending correct actions to robot from raw network predictions.
        """
        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -0.25, 0.25

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == "TCP_position_control":

            max_pos_change = 0.001  # m per step
            max_ang_change = 1 * (np.pi / 180)  # rad per step

            self.x_act_min, self.x_act_max = -self.x_speed_ratio*max_pos_change, self.x_speed_ratio*max_pos_change
            self.y_act_min, self.y_act_max = -self.y_speed_ratio*max_pos_change, self.y_speed_ratio*max_pos_change
            self.z_act_min, self.z_act_max = -0, 0
            self.roll_act_min, self.roll_act_max = -0, 0
            self.pitch_act_min, self.pitch_act_max = -0, 0
            self.yaw_act_min, self.yaw_act_max = -self.Rz_speed_ratio*max_ang_change, self.Rz_speed_ratio*max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -0, 0
            self.roll_act_min, self.roll_act_max = -0, 0
            self.pitch_act_min, self.pitch_act_max = -0, 0
            self.yaw_act_min, self.yaw_act_max = -max_ang_vel, max_ang_vel

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_rgb_obs_camera_params(self):
        self.rgb_cam_pos = [0.15, 0.0, -0.35]
        self.rgb_cam_dist = 1.0
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -45
        self.rgb_image_size = self._image_size
        # self.rgb_image_size = [512,512]
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

    def setup_object(self):
        """
        Set vars for loading an object
        """
        # currently hardcode these for cube, could pull this from bounding box
        self.obj_width = 0.08
        self.obj_height = 0.08

        # define an initial position for the objects (world coords)
        self.init_obj_pos = self.workframe_pos + np.array([0.0, self.obj_width / 2 - 0.001, 0])
        self.init_obj_rpy =  np.array([-np.pi, 0.0, np.pi / 2])
        self.init_obj_orn = self._pb.getQuaternionFromEuler(self.init_obj_rpy)

        # get paths
        self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_push/cube/cube.urdf")

        self.goal_path = add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf")

    def reset_object(self, init_orn=None, init_pos_y = None):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        # reser the position of the object
        if self.rand_init_orn and (not self.eval_mode):
            if init_orn:
                self.init_obj_ang = init_orn
            else:
                # self.init_obj_ang = self.np_random.uniform(-np.pi / 32, np.pi / 32)
                self.init_obj_ang = self.np_random.uniform(-np.deg2rad(20), np.deg2rad(20))      
        else:
            self.init_obj_ang = 0.0

        self.init_obj_orn = self._pb.getQuaternionFromEuler(
            [-np.pi, 0.0, np.pi / 2 + self.init_obj_ang]
        )

        if self.rand_init_pos_y and (not self.eval_mode):
            if init_pos_y:
                self.init_obj_y =  init_pos_y
            else:
                self.init_obj_y = self.np_random.uniform(-self.obj_width/2*0.5, self.obj_width/2*0.5)
        else:
            self.init_obj_y = 0.0

        # Calculate new postion and orientation of object after randomised initialisation
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.init_obj_orn)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        x_vector = np.array([1, 0, 0])
        y_vector = np.array([0, 1, 0])
        obj_x_vector = obj_rot_matrix.dot(x_vector)
        obj_y_vector = obj_rot_matrix.dot(y_vector)
        contact = self.init_obj_pos - self.obj_width/2 * obj_x_vector + self.init_obj_y * obj_y_vector
        delta_to_contact = self.workframe_pos - contact

        self._pb.resetBasePositionAndOrientation(
            self.obj_id, self.init_obj_pos + delta_to_contact, self.init_obj_orn
        )

        # perform object dynamics randomisations
        self._pb.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=0.065,
            spinningFriction=0.00,
            rollingFriction=0.00,
            restitution=0.0,
            frictionAnchor=1,
            collisionMargin=0.0001,
        )

        if self.rand_obj_mass and (not self.eval_mode):
            obj_mass = self.np_random.uniform(0.4, 0.8)
            self._pb.changeDynamics(self.obj_id, -1, mass=obj_mass)

    def load_trajectory(self):

        self.traj_n_points = 1
        self.traj_spacing = 0.05
        self.traj_max_perturb = 0.2

        # place goals at each point along traj
        self.traj_ids = []
        for i in range(int(self.traj_n_points)):
            pos = [0.0, 0.0, 0.0]
            traj_point_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                pos,
                [0, 0, 0, 1],
                useFixedBase=True,
            )
            self._pb.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(traj_point_id, -1, 0, 0)
            self.traj_ids.append(traj_point_id)

        # DEBUG: Load a object marker
        self.obj_marker_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                pos,
                [0, 0, 0, 1],
                useFixedBase=True,
            )
        self._pb.changeVisualShape(self.obj_marker_id, -1, rgbaColor=[0, 1, 1, 1])
        self._pb.setCollisionFilterGroupMask(self.obj_marker_id, -1, 0, 0)

    def update_trajectory(self, single_goal=np.array([])):

        # setup traj arrays
        self.targ_traj_list_id = -1
        self.traj_pos_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_rpy_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_orn_workframe = np.zeros(shape=(self.traj_n_points, 4))


        # Reset goals based on deterministic goals for evluation environment
        if self.eval_mode and not np.any(single_goal): 
            self.update_trajectory_point(self.eval_goals[self.eval_goal_index])

            self.eval_goal_index += 1

            # Reset eval goals to the beginning of the list
            if self.eval_goal_index >= self.eval_num:
                self.eval_goal_index = 0

        else:
            self.update_trajectory_point(single_goal)
        
        # generate initial orientation to place object at
        self.init_obj_pos_workframe, self.init_obj_rpy_workframe = self.robot.arm.worldframe_to_workframe(
                                                                        self.init_obj_pos, self.init_obj_rpy)
        self.init_obj_orn_workframe = self._pb.getQuaternionFromEuler(self.init_obj_rpy_workframe)
        self.traj_rpy_workframe[:, 2] = np.arctan2(-self.init_obj_pos_workframe[1] + self.traj_pos_workframe[:, 1], 
                                                    -self.init_obj_pos_workframe[0] + self.traj_pos_workframe[:, 0])

        # Get goal in work frame
        for i in range(int(self.traj_n_points)):
            # get workframe orn
            self.traj_orn_workframe[i] = self._pb.getQuaternionFromEuler(
                self.traj_rpy_workframe[i]
            )

            # convert worldframe
            pos_worldframe, rpy_worldframe = self.robot.arm.workframe_to_worldframe(
                self.traj_pos_workframe[i], self.traj_rpy_workframe[i]
            )
            orn_worldframe = self._pb.getQuaternionFromEuler(rpy_worldframe)

            # place goal
            self._pb.resetBasePositionAndOrientation(
                self.traj_ids[i], pos_worldframe, orn_worldframe
            )
            self._pb.changeVisualShape(self.traj_ids[i], -1, rgbaColor=[0, 1, 0, 0.5])

        # reward normalisers for starting pos and orn at 0
        self.pos_reward_normaliser = np.linalg.norm(self.traj_pos_workframe[0])
        self.orn_reward_normaliser = np.arccos(np.clip((2 * self.traj_orn_workframe[0, 3] ** 2) - 1, -1, 1))
        self.orn_tip_obj_normaliser = self.max_tcp_to_obj_orn

    
    def random_single_goal(self, axis=None):

        # create a random single goal that is at least 0.2m away from the object initial position
        if axis:
            random_axis = axis
        else:
            if len(self.goal_edges) > 1:
                random_axis_idx = self.np_random.choice(range(len(self.goal_edges)))
                random_axis = self.goal_edges[int(random_axis_idx)]
            else:
                random_axis = self.goal_edges[0]

        # Randomly sample goals that are close to the initial position
        p = np.random.random()
        x_factor = 1.0
        y_factor = 1.0

        # random x-axis
        if random_axis[0] == 0:
            if random_axis[1] == -1:
                y = self.goal_y_min
            else:
                y = self.goal_y_max
            random_x = self.np_random.uniform(low=self.goal_x_min, high=self.goal_x_max)
            # random_x = x_min
            # random_x = 0
            return (random_x * x_factor, y * y_factor)
        # random y-axis
        else:
            if random_axis[0] == -1:
                x = self.goal_x_min
            else:
                x = self.goal_x_max
            random_y = self.np_random.uniform(self.goal_y_min, self.goal_y_max)
            # random_y = 0
            return (x * x_factor, random_y * y_factor)

    def make_n_point_goals(self, num_trials):

        # Create evenly distributed goals along the edge
        n_point_per_side, n_random = divmod(num_trials, len(self.goal_edges))

        n_points = n_point_per_side * np.ones(len(self.goal_edges), dtype=int)

        # work out the number of duplicate goals
        num_duplicates = 0
        if len(self.goal_edges) == 3:
            num_duplicates = 2
        elif len(self.goal_edges) == 4:
            num_duplicates = 4
        n_random += num_duplicates

        # Factor in duplicate goals into points per sides
        count = 0
        for i in range(n_random):
            n_points[count] += 1
            count += 1
            if count >= len(n_points):
                count = 0

        evaluation_goals = np.array([])
        for i, edge in enumerate(self.goal_edges):
            # random x-axis
            goal_edges = np.zeros((n_points[i], 2))
            if edge[0] == 0:
                if edge[1] == -1:
                    y = self.goal_y_min
                else:
                    y = self.goal_y_max
                x = np.linspace(self.goal_x_min, self.goal_x_max, num=n_points[i])
            # random y axis
            else:
                if edge[0] == -1:
                    x = self.goal_x_min
                else:
                    x = self.goal_x_max
                y = np.linspace(self.goal_y_min, self.goal_y_max, num=n_points[i])
            goal_edges[:, 0] = x
            goal_edges[:, 1] = y

            evaluation_goals = np.hstack([
                *evaluation_goals,
                *goal_edges
            ])

        # get unique goals
        evaluation_goals = evaluation_goals.reshape(sum(n_points), 2)
        evaluation_goals = np.unique(evaluation_goals,axis=0)

        return evaluation_goals

    def update_trajectory_point(self, single_goal=np.array([])):

        # Create a single trajectory goal
        # Check if empty 
        if not np.any(single_goal):
            (x, y) = self.random_single_goal()
        else:
            (x, y) = single_goal
        z = 0.0
        self.traj_pos_workframe[0] = [x, y, z]

    def make_goal(self, single_goal=np.array([])):
        """
        Generate a goal place a set distance from the inititial object pose.
        """
        # update the curren trajecory
        self.update_trajectory(single_goal)

        # set goal as first point along trajectory
        self.update_goal()
        
    def update_goal(self):
        """
        move goal along trajectory
        """
        # increment targ list
        self.targ_traj_list_id += 1

        if self.targ_traj_list_id >= self.traj_n_points:
            return False
        else:
            self.goal_id = self.traj_ids[self.targ_traj_list_id]

            # get goal pose in world frame
            (
                self.goal_pos_worldframe,
                self.goal_orn_worldframe,
            ) = self._pb.getBasePositionAndOrientation(self.goal_id)
            self.goal_rpy_worldframe = self._pb.getEulerFromQuaternion(
                self.goal_orn_worldframe
            )

            # create variables for goal pose in workframe to use later
            self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
            self.goal_orn_workframe = self.traj_orn_workframe[self.targ_traj_list_id]
            self.goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id]

            # change colour of new target goal
            self._pb.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 0, 1, 0.5])

            # change colour of goal just reached
            prev_goal_traj_list_id = (
                self.targ_traj_list_id - 1 if self.targ_traj_list_id > 0 else None
            )
            if prev_goal_traj_list_id is not None:
                self._pb.changeVisualShape(
                    self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5]
                )

            return True

    def encode_TCP_frame_actions(self, actions):

        encoded_actions = np.zeros(6)

        # get rotation matrix from current tip orientation
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        # define initial vectors
        par_vector = np.array([1, 0, 0])  # outwards from tip
        perp_vector = np.array([0, -1, 0])  # perp to tip

        # find the directions based on initial vectors
        par_tip_direction = tip_rot_matrix.dot(par_vector)
        perp_tip_direction = tip_rot_matrix.dot(perp_vector)

        # transform into workframe frame for sending to robot
        workframe_par_tip_direction = self.robot.arm.worldvec_to_workvec(
            par_tip_direction
        )
        workframe_perp_tip_direction = self.robot.arm.worldvec_to_workvec(
            perp_tip_direction
        )
        
        # translate the direction
        perp_scale = actions[1]
        perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

        par_scale = actions[0]
        par_action = np.dot(workframe_par_tip_direction, par_scale)

        encoded_actions[0] += perp_action[0] + par_action[0]
        encoded_actions[1] += perp_action[1] + par_action[1]
        encoded_actions[5] += actions[5]

        return encoded_actions

    def encode_work_frame_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm.
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "y":
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]

        if self.movement_mode == "yRz":
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]
            encoded_actions[5] = actions[1]

        elif self.movement_mode == "xyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[5] = actions[2]

        return encoded_actions

    def encode_actions(self, actions):
        # scale and embed actions appropriately
        if self.movement_mode in ["y", "yRz", "xyRz"]:
            encoded_actions = self.encode_work_frame_actions(actions)
        elif self.movement_mode in ["TyRz", "TxTyRz"]:
            encoded_actions = self.encode_TCP_frame_actions(actions)
        return encoded_actions

    def get_step_data(self, action=np.zeros(2)):

        # get the cur tip pos here for once per step
        (
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_rpy_worldframe,
            self.cur_tcp_orn_worldframe,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()
        (
            self.cur_obj_pos_worldframe,
            self.cur_obj_orn_worldframe,
        ) = self.get_obj_pos_worldframe()
        (
            self.cur_obj_center_worldframe,
            _,
        ) = self.get_obj_center_worldframe()
        self.cur_obj_rpy_worldframe = self._pb.getEulerFromQuaternion(self.cur_obj_orn_worldframe)
        
        # Update goal orn outside the sensitive regions
        self.update_goal_orn()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward(action)

        # get rl info
        done, reward = self.termination(reward)

        #### DEBUG: Update position of the object marker
        obj_marker_worldframe = self.cur_obj_pos_worldframe
        obj_marker_worldframe = np.array(obj_marker_worldframe)
        obj_marker_worldframe[2] += self.obj_width*2

        self._pb.resetBasePositionAndOrientation(
            self.obj_marker_id, obj_marker_worldframe, [0, 0, 0, 1]
        )
        self._pb.changeVisualShape(self.obj_marker_id, -1, rgbaColor=[0, 1, 1, 1])

        line_scale = 0.2
        start_point = obj_marker_worldframe
        normal = np.array([0, 0, -1]) * line_scale
        self._pb.addUserDebugLine(start_point, start_point + normal, [0, 1, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        
        #### DEBUG

        return reward, done

    def update_goal_orn(self):

        # get goal point in obj frame
        goal_pos_objframe, _ = self.worldframe_to_objframe(self.goal_pos_worldframe, np.zeros(3))
        obj_Rz_to_goal_objframe = np.arctan2(goal_pos_objframe[1],  goal_pos_objframe[0])
        cur_obj_rpy_to_goal_objframe = np.array([0.0, 0.0, obj_Rz_to_goal_objframe])

        # Convert into worldframe
        _, self.goal_rpy_worldframe = self.objframe_to_worldframe(np.zeros(3), cur_obj_rpy_to_goal_objframe)
        self.goal_orn_worldframe = np.array(self._pb.getQuaternionFromEuler(self.goal_rpy_worldframe))

        # Convert into workframe
        _, self.goal_rpy_workframe = self.robot.arm.worldframe_to_workframe(np.zeros(3), self.goal_rpy_worldframe)
        self.goal_orn_workframe = np.array(self._pb.getQuaternionFromEuler(self.goal_rpy_workframe))

    def termination(self, reward):
        """
        Criteria for terminating an episode.
        """

        # check if near goal 
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        
        if obj_goal_pos_dist < self.termination_pos_dist:

            # update the goal (if not at end of traj)
            self.goal_updated = self.update_goal()

            if not self.goal_updated:
                self.single_goal_reached = True
                # print("Goal reached")

                return True, reward
        else:
            self.goal_updated = False

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True, reward

        # check if lost contact: if losing contact in consecutive steps, then end episode
        if self.use_contact:
            if (not self.contact_obs_after_action) and (self._env_step_counter != 0):
                self.lost_contact_count += 1
                if self.lost_contact_count >= self.max_no_contact_steps:
                    reward += self.terminated_early_penalty

                    # Contact is loss, return termination results
                    if self.terminate_terminate_early:
                        return True, reward
                    else:
                        return False, reward
            else:
                self.lost_contact_count = 0

        # Terminate early when object or TCP moves out of TCP_lims
        if self.terminate_early and self.early_termination():
            reward += self.terminated_early_penalty
            if self.terminate_terminate_early:
                return True, reward
            else:
                return False, reward

        return False, reward

    def early_termination(self):

        # Terminate if object or tcp leaves the tcp limits
        (
            tcp_pos_workframe,
            _,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()
        cur_obj_pos_workframe, _ = self.get_obj_pos_workframe()

        abs_tcp_to_obj_orn = self.orn_tcp_dist_to_obj()

        TCP_lim_tol = 0.001
        if ((tcp_pos_workframe[0] <= self.robot.arm.TCP_lims[0,0] + TCP_lim_tol) or 
        (tcp_pos_workframe[0] >= self.robot.arm.TCP_lims[0,1] - TCP_lim_tol) or 
        (tcp_pos_workframe[1] <= self.robot.arm.TCP_lims[1,0] + TCP_lim_tol) or 
        (tcp_pos_workframe[1] >= self.robot.arm.TCP_lims[1,1] - TCP_lim_tol) or 
        (cur_obj_pos_workframe[0] <= self.robot.arm.TCP_lims[0,0] + TCP_lim_tol) or 
        (cur_obj_pos_workframe[0] >= self.robot.arm.TCP_lims[0,1] - TCP_lim_tol) or 
        (cur_obj_pos_workframe[1] <= self.robot.arm.TCP_lims[1,0] + TCP_lim_tol) or 
        (cur_obj_pos_workframe[1] >= self.robot.arm.TCP_lims[1,1] - TCP_lim_tol) or 
        (abs_tcp_to_obj_orn >= self.max_tcp_to_obj_orn)):
            return True

        return False

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        +1 is given for each goal reached.
        This is calculated before termination called as that will update the goal.
        """
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        if obj_goal_pos_dist < self.termination_pos_dist:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def dense_reward(self, action):
        """
        Calculate the reward when in dense mode.
        """
        # Assign weights for each rewards
        self.W_obj_goal_pos = 0.0
        self.W_obj_goal_orn = self.importance_obj_goal_orn
        self.W_tip_obj_orn = self.importance_tip_obj_orn

        if (self.xyz_obj_dist_to_goal() <= 0.1):
            self.W_obj_goal_pos = self.importance_obj_goal_pos 
            self.W_obj_goal_orn = 0.0
            self.W_tip_obj_orn = self.importance_obj_goal_pos / 5

        # sum rewards with multiplicative factors
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        tip_obj_orn_dist = self.orn_tcp_dist_to_obj()
        
        reward = -(
            (self.W_obj_goal_pos * obj_goal_pos_dist)
            + (self.W_obj_goal_orn * obj_goal_orn_dist)
            + (self.W_tip_obj_orn * tip_obj_orn_dist)
        )

        return reward

    def get_goal_obs(self):
        '''
        Return goal obs for each corresponding observation mode
        '''
        if self.observation_mode == "tactile_pose_array":
            goals = self.get_tactile_pose_goals()
        else:
            raise ValueError('Compute goal-based reward called for non goal-based observation mode')

        return goals
    
    def get_orn_to_goal_workframe(self, cur_orn_workframe):
        """
        Calculate the difference between an orientaion quaternion and the goal 
        orientation in the workframe
        """

        return get_orn_diff(cur_orn_workframe, self.goal_orn_workframe)
    
    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on object
        cur_obj_pos_workframe, cur_obj_orn_workframe = self.get_obj_pos_workframe()
        cur_obj_rpy_workframe = self._pb.getEulerFromQuaternion(cur_obj_orn_workframe)
        (
            cur_obj_lin_vel_workframe,
            cur_obj_ang_vel_workframe,
        ) = self.get_obj_vel_workframe()

        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            tcp_orn_workframe,
            tcp_lin_vel_workframe,
            tcp_ang_vel_workframe,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # stack into array
        observation = np.hstack(
            [
                *tcp_pos_workframe,
                *tcp_rpy_workframe,
                *tcp_lin_vel_workframe,
                *tcp_ang_vel_workframe,
                *cur_obj_pos_workframe,
                *cur_obj_rpy_workframe,
                *cur_obj_lin_vel_workframe,
                *cur_obj_ang_vel_workframe,
                *self.goal_pos_workframe,
                *self.goal_rpy_workframe,
            ]
        )

        return observation
    
    def get_goal_aware_tactile_pose_obs(self):
        """
        Use for sanity checking, tactile pose information that can be 
        measured using the tactip.
        """

        # get sim info on object 
        cur_obj_pos_workframe, cur_obj_orn_workframe = self.get_obj_pos_workframe()
        cur_obj_rpy_workframe = self._pb.getEulerFromQuaternion(cur_obj_orn_workframe)

        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # Calculate relevant states for tcp
        tcp_pos_to_obj_workframe = tcp_pos_workframe - cur_obj_pos_workframe
        tcp_Rz_to_obj_workframe = tcp_rpy_workframe[2] - cur_obj_rpy_workframe[2]

        tcp_rpy_to_obj_workframe = np.array([0.0, 0.0, tcp_Rz_to_obj_workframe])
        tcp_orn_to_obj_workframe = self._pb.getQuaternionFromEuler(tcp_rpy_to_obj_workframe)
    
        # Calculate relevant states for object
        cur_obj_pos_to_goal_workframe = cur_obj_pos_workframe - self.goal_pos_workframe
        cur_obj_Rz_to_goal_workframe = cur_obj_rpy_workframe[2] - self.goal_rpy_workframe[2]  

        cur_obj_rpy_to_goal_workframe = np.array([0.0, 0.0, cur_obj_Rz_to_goal_workframe])
        cur_obj_orn_to_goal_workframe = self._pb.getQuaternionFromEuler(cur_obj_rpy_to_goal_workframe)

        # stack into array
        observation = np.hstack(
            [
                *(tcp_pos_to_obj_workframe[0:2]),
                *(tcp_orn_to_obj_workframe[2:4]),
                *(cur_obj_pos_to_goal_workframe[0:2]),
                *(cur_obj_orn_to_goal_workframe[2:4]),
            ]
        )

        return observation

    def get_tactile_pose_obs(self):

        # get sim info on object 
        cur_obj_pos_workframe, cur_obj_orn_workframe = self.get_obj_pos_workframe()
        cur_obj_rpy_workframe = self._pb.getEulerFromQuaternion(cur_obj_orn_workframe)

        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # Calculate relevant states for tcp
        tcp_pos_to_obj_workframe = tcp_pos_workframe - cur_obj_pos_workframe
        tcp_Rz_to_obj_workframe = tcp_rpy_workframe[2] - cur_obj_rpy_workframe[2]

        tcp_rpy_to_obj_workframe = np.array([0.0, 0.0, tcp_Rz_to_obj_workframe])
        tcp_orn_to_obj_workframe = self._pb.getQuaternionFromEuler(tcp_rpy_to_obj_workframe)


        # stack into array
        observation = np.hstack(
            [
                *(tcp_pos_to_obj_workframe[0:2]),
                *(tcp_orn_to_obj_workframe[2:4]),
                *(cur_obj_pos_workframe[0:2]),
                *(cur_obj_orn_workframe[2:4]),
            ]
        )

        return observation

    def get_tactile_pose_goals(self):
        """
        Return goal state for corresponding tactile pose observation
        """

        observation = np.hstack(
            [
                *(np.array([0, 0])),
                *(np.array([0, 1])),
                *self.goal_pos_workframe[0:2],
                *self.goal_orn_workframe[2:4],
            ]
        )

        return observation

    def get_obs_workframe(self):
        """
        Return tcp and object observation in the workframe for validation
        """
        cur_obj_pos_workframe, cur_obj_orn_workframe = self.get_obj_pos_workframe()
        cur_obj_rpy_workframe = self._pb.getEulerFromQuaternion(cur_obj_orn_workframe)

          # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        return tcp_pos_workframe, tcp_rpy_workframe, cur_obj_pos_workframe, cur_obj_rpy_workframe

    def get_extended_feature_array(self):
        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # Get relative observation
        tcp_pos_to_goal_workframe = tcp_pos_workframe - self.goal_pos_workframe
        tcp_Rz_to_goal_workframe = tcp_rpy_workframe[2] - self.goal_rpy_workframe[2]

        tcp_rpy_to_goal_workframe = np.array([0.0, 0.0, tcp_Rz_to_goal_workframe])
        tcp_orn_to_goal_workframe = self._pb.getQuaternionFromEuler(tcp_rpy_to_goal_workframe)

        feature_array = np.array(
            [
                *tcp_pos_to_goal_workframe[0:2],
                *tcp_orn_to_goal_workframe[2:4],
            ]
        )

        return feature_array

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        if self.movement_mode == "y":
            return 1
        if self.movement_mode == "yRz":
            return 2
        if self.movement_mode == "xyRz":
            return 3
        if self.movement_mode == "TyRz":
            return 2
        if self.movement_mode == "TxTyRz":
            return 3