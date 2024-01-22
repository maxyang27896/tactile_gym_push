import os, sys
from collections import deque
import time
import gym
from gym import utils
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pkgutil
import cv2

from tactile_gym.assets import get_assets_path, add_assets_path

class BaseTactileEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        max_steps=250,
        image_size=[64, 64],
        obs_stacked_len=1,
        show_gui=False,
        show_tactile=False,
    ):

        # set seed
        self.seed()

        # env params
        self._observation = []
        self._env_step_counter = 0
        self._max_steps = max_steps
        self._image_size = image_size
        self.obs_history_len = obs_stacked_len
        self._show_gui = show_gui
        self._show_tactile = show_tactile
        self._first_render = True
        self._render_closed = False
        self.contact_obs_after_action = None
        self.lost_contact_count = 0
        self.last_contact_pos = None

        # set up camera for rgb obs and debbugger
        self.setup_rgb_obs_camera_params()

        self.connect_pybullet()

        # set vars for full pybullet reset to clear cache
        self.reset_counter = 0
        self.reset_limit = 1000

        utils.EzPickle.__init__(self)

    def connect_pybullet(self):
        # render the environment
        if self._show_gui:
            self._pb = bc.BulletClient(connection_mode=pb.GUI)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(
                self._pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0
            )
            self._pb.configureDebugVisualizer(
                self._pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0
            )
            self._pb.resetDebugVisualizerCamera(
                self.rgb_cam_dist,
                self.rgb_cam_yaw,
                self.rgb_cam_pitch,
                self.rgb_cam_pos,
            )
        else:
            self._pb = bc.BulletClient(connection_mode=pb.DIRECT)
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                self._pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self._pb.loadPlugin("eglRendererPlugin")

        # bc automatically sets client but keep here incase needed
        self._physics_client_id = self._pb._client

    def seed(self, seed=None):
        self._seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self.close()

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()

        if not self._render_closed:
            cv2.destroyAllWindows()

    def setup_observation_space(self):

        obs_dim_dict = self.get_obs_dim()

        spaces = {
            'oracle':gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict['oracle'], dtype=np.float32
            ),
            'tactile_pose':gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict['tactile_pose'], dtype=np.float32
            ),
            'goal_aware_tactile_pose':gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict['goal_aware_tactile_pose'], dtype=np.float32
            ),
            'tactile': gym.spaces.Box(
                low=0, high=255, shape=obs_dim_dict['tactile'], dtype=np.uint8
            ),
            'visual': gym.spaces.Box(
                low=0, high=255, shape=obs_dim_dict['visual'], dtype=np.uint8
            ),
            'extended_feature': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict['extended_feature'], dtype=np.float32
            )
        }

        if self.observation_mode == "oracle":
            self.observation_space = gym.spaces.Dict({
                'oracle': spaces['oracle']
            })
        elif self.observation_mode == "goal_aware_tactile_pose":
            self.observation_space = gym.spaces.Dict({
                'oracle': spaces['goal_aware_tactile_pose']
            })

        # For MBRB LIB entries        
        elif self.observation_mode == "tactile_pose_array":
            self.observation_space = spaces['tactile_pose']


        elif self.observation_mode == "tactile":
            self.observation_space = gym.spaces.Dict({
                'tactile': spaces['tactile']
            })

        elif self.observation_mode == "visual":
            self.observation_space = gym.spaces.Dict({
                'visual': spaces['visual']
            })

        elif self.observation_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({
                'tactile': spaces['tactile'],
                'visual': spaces['visual']
            })

        elif self.observation_mode == "tactile_and_feature":
            self.observation_space = gym.spaces.Dict({
                'tactile': spaces['tactile'],
                'extended_feature': spaces['extended_feature']
            })

        elif self.observation_mode == "visual_and_feature":
            self.observation_space = gym.spaces.Dict({
                'visual': spaces['visual'],
                'extended_feature': spaces['extended_feature']
            })

        elif self.observation_mode == "visuotactile_and_feature":
            self.observation_space = gym.spaces.Dict({
                'tactile': spaces['tactile'],
                'visual': spaces['visual'],
                'extended_feature': spaces['extended_feature']
            })        

    def get_obs_dim(self):
        obs_dim_dict = {
            'oracle': self.get_oracle_obs().shape,
            'tactile_pose': self.get_tactile_pose_obs().shape,
            'goal_aware_tactile_pose': self.get_goal_aware_tactile_pose_obs().shape,
            'tactile': self.get_tactile_obs().shape,
            'visual': self.get_visual_obs().shape,
            'extended_feature': self.get_extended_feature_array().shape,
        }
        return obs_dim_dict

    def load_environment(self):

        self._pb.setGravity(0, 0, -9.81)
        self._pb.setPhysicsEngineParameter(fixedTimeStep=self._sim_time_step,
                                           numSolverIterations=150,
                                           enableConeFriction=1,
                                           contactBreakingThreshold=0.0001
                                           )
        self.plane_id = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/plane/plane.urdf"),
            [0, 0, -0.625],
        )
        self.table_id = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/table/table.urdf"),
            [0.50, 0.00, -0.625],
            [0.0, 0.0, 0.0, 1.0],
        )

    def scale_actions(self, actions):

        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, self.min_action, self.max_action)

        input_range = self.max_action - self.min_action

        new_x_range = self.x_act_max - self.x_act_min
        new_y_range = self.y_act_max - self.y_act_min
        new_z_range = self.z_act_max - self.z_act_min
        new_roll_range = self.roll_act_max - self.roll_act_min
        new_pitch_range = self.pitch_act_max - self.pitch_act_min
        new_yaw_range = self.yaw_act_max - self.yaw_act_min

        scaled_actions = [
            (((actions[0] - self.min_action) * new_x_range) / input_range)
            + self.x_act_min,
            (((actions[1] - self.min_action) * new_y_range) / input_range)
            + self.y_act_min,
            (((actions[2] - self.min_action) * new_z_range) / input_range)
            + self.z_act_min,
            (((actions[3] - self.min_action) * new_roll_range) / input_range)
            + self.roll_act_min,
            (((actions[4] - self.min_action) * new_pitch_range) / input_range)
            + self.pitch_act_min,
            (((actions[5] - self.min_action) * new_yaw_range) / input_range)
            + self.yaw_act_min,
        ]

        return np.array(scaled_actions)

    def scale_tcp_actions(self, actions):

        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, self.min_action, self.max_action)

        input_actions = np.zeros(6)
        if self.movement_mode == "TyRz":
            input_actions[0] = self.max_action
            input_actions[1] = actions[0] 
            input_actions[5] = actions[1] 
        elif self.movement_mode == "TxTyRz":
            input_actions[0] = actions[0] 
            input_actions[1] = actions[1] 
            input_actions[5] = actions[2] 

        input_range = self.max_action - self.min_action

        new_x_range = self.x_act_max - self.x_act_min
        new_y_range = self.y_act_max - self.y_act_min
        new_z_range = self.z_act_max - self.z_act_min
        new_roll_range = self.roll_act_max - self.roll_act_min
        new_pitch_range = self.pitch_act_max - self.pitch_act_min
        new_yaw_range = self.yaw_act_max - self.yaw_act_min

        scaled_actions = [
            (((input_actions[0] - self.min_action) * new_x_range) / input_range)
            + self.x_act_min,
            (((input_actions[1] - self.min_action) * new_y_range) / input_range)
            + self.y_act_min,
            (((input_actions[2] - self.min_action) * new_z_range) / input_range)
            + self.z_act_min,
            (((input_actions[3] - self.min_action) * new_roll_range) / input_range)
            + self.roll_act_min,
            (((input_actions[4] - self.min_action) * new_pitch_range) / input_range)
            + self.pitch_act_min,
            (((input_actions[5] - self.min_action) * new_yaw_range) / input_range)
            + self.yaw_act_min,
        ]
    
        return np.array(scaled_actions)

    def step(self, action, count_step=True):

        # scale and embed actions appropriately
        if self.movement_mode in ["TyRz", "TxTyRz"]:
            scaled_actions = self.scale_tcp_actions(action)
            input_actions = self.encode_actions(scaled_actions)
        else:
            encoded_actions = self.encode_actions(action)
            input_actions = self.scale_actions(encoded_actions)
            
        if count_step:
            self._env_step_counter += 1

        pybullet_steps, self.contact_obs_after_action = self.robot.apply_action(
            input_actions,
            control_mode=self.control_mode,
            velocity_action_repeat=self._velocity_action_repeat,
            max_steps=self._max_blocking_pos_move_steps,
        )
            
        reward, done = self.get_step_data(action)

        self._observation = self.get_observation()

        is_contact = True if self.contact_obs_after_action else False

        # can be helpful when debugging
        # self.render()
        # time.sleep(0.01)

        return self._observation, reward, done, {"num_of_pb_steps": pybullet_steps, "tip_in_contact": is_contact}

    def is_tip_in_contact(self, contacts=None):
        """
        Return true or false if tactip is in contact with object
        """
        return None

    def get_extended_feature_array(self):
        """
        Get feature to extend current observations relative to goal.
        """
        return np.array([])

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        return np.array([])

    def get_tactile_pose_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        return np.array([])

    def get_goal_aware_tactile_pose_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        return np.array([])

    def get_tactile_obs(self):
        """
        Returns the tactile observation with an image channel.
        """

        # get image from sensor
        tactile_obs = self.robot.get_tactile_observation()

        observation = tactile_obs[..., np.newaxis]

        return observation

    def is_tip_in_contact(self, contacts=None):
        """
        Return true or false if tactip is in contact with object
        """
        return None

    def get_visual_obs(self):
        """
        Returns the rgb image from an environment camera.
        """
        # get an rgb image that matches the debug visualiser
        view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.rgb_cam_pos,
            distance=self.rgb_cam_dist,
            yaw=self.rgb_cam_yaw,
            pitch=self.rgb_cam_pitch,
            roll=0,
            upAxisIndex=2,
        )

        proj_matrix = self._pb.computeProjectionMatrixFOV(
            fov=self.rgb_fov,
            aspect=float(self.rgb_image_size[0]) / self.rgb_image_size[1],
            nearVal=self.rgb_near_val,
            farVal=self.rgb_far_val,
        )

        (_, _, px, _, _) = self._pb.getCameraImage(
            width=self.rgb_image_size[0],
            height=self.rgb_image_size[1],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(
            rgb_array, (self.rgb_image_size[0], self.rgb_image_size[1], 4)
        )

        observation = rgb_array[:, :, :3]

        return observation

    def get_observation(self):
        """
        Returns the observation dependent on which mode is set.
        """
        # init obs dict
        observation = {}

        # check correct obs type set
        if self.observation_mode not in [
            'oracle',
            'goal_aware_tactile_pose',      # model-free obs
            'tactile_pose_array',           # model-based obs

            # Use for tactile images
            'tactile',
            'visual',
            'visuotactile',
            'tactile_and_feature',
            'visual_and_feature',
            'visuotactile_and_feature',
        ]:
            sys.exit(
                "Incorrect observation mode specified: {}".format(self.observation_mode)
            )

        if self.observation_mode == "tactile_pose_array":
            return self.get_tactile_pose_obs()

        # use direct pose info to check if things are working
        if "oracle" in self.observation_mode:
            observation['oracle'] = self.get_oracle_obs()
        
        # observation is just the tactile sensor image
        if "tactile" in self.observation_mode:
            if self.observation_mode == "goal_aware_tactile_pose":
                observation['oracle'] = self.get_goal_aware_tactile_pose_obs()
            else:
                observation['tactile'] = self.get_tactile_obs()

        # observation is rgb environment camera image
        if any(obs in self.observation_mode for obs in ["visual", "visuo"]):
            observation['visual'] = self.get_visual_obs()

        # observation is mix image + features (pretending to be image shape)
        if "feature" in self.observation_mode:
            observation['extended_feature'] = self.get_extended_feature_array()

        return observation

    def render(self, mode="rgb_array"):
        """
        Most rendering handled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_obs()
        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # resize tactile to match rgb if rendering in higher res
        if self._image_size != self.rgb_image_size:
            tactile_array = cv2.resize(tactile_array, tuple(self.rgb_image_size))

        # concat the images into a single image
        render_array = np.concatenate([rgb_array, tactile_array], axis=1)

        # setup plot for rendering
        if self._first_render:
            self._first_render = False
            if self._seed is not None:
                self.window_name = 'render_window_{}'.format(self._seed)
            else:
                self.window_name = 'render_window'
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # plot rendered image
        if not self._render_closed:
            render_array_rgb = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
            cv2.imshow(self.window_name, render_array_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(self.window_name)
                self._render_closed = True

        return render_array
