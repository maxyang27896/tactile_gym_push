from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)

import sys, os, ctypes
import numpy as np

class RedirectStream(object):

  @staticmethod
  def _flush_c_stream(stream):
    streamname = stream.name[1:-1]
    libc = ctypes.CDLL(None)
    libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

  def __init__(self, stream=sys.stdout, file=os.devnull):
    self.stream = stream
    self.file = file

  def __enter__(self):
    self.stream.flush()  # ensures python stream unaffected 
    self.fd = open(self.file, "w+")
    self.dup_stream = os.dup(self.stream.fileno())
    os.dup2(self.fd.fileno(), self.stream.fileno()) # replaces stream
  
  def __exit__(self, type, value, traceback):
    RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
    os.dup2(self.dup_stream, self.stream.fileno()) # restores stream
    os.close(self.dup_stream)
    self.fd.close()

def main():

    seed = int(0)
    num_iter = 10
    max_steps = 10000
    show_gui = True
    show_tactile = False
    render = False
    print_info = False
    image_size = [256, 256]
    env_modes = {
        ## which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        # "movement_mode": "xyRz",
        'movement_mode':'TyRz',
        # 'movement_mode':'TxTyRz',

        ## the type of control used
        'control_mode':'TCP_position_control',
        # "control_mode": "TCP_velocity_control",

        ## randomisations
        "rand_init_orn": False,
        "rand_init_pos_y": False,
        "rand_obj_mass": False,

        ## straight or random trajectory
        # "traj_type": "straight",
        'traj_type':'point',

        ## which observation type to return
        # 'observation_mode':'oracle',
        # 'observation_mode':'oracle_reduced',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
        'observation_mode':'tactile_pose_goal_excluded',

        'terminate_early': True,
        'use_contact': True,
        'task': 'goal_pos',


        ## the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    # set limits and goals
    TCP_lims = np.zeros(shape=(6, 2))
    TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, 0.3  # x lims
    TCP_lims[1, 0], TCP_lims[1, 1] = -0.3, 0.3  # y lims
    TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
    TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
    TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
    TCP_lims[5, 0], TCP_lims[5, 1] = -180 * np.pi / 180, 180 * np.pi / 180  # yaw lims

    # goal parameter
    goal_edges = ((0, -1), (0, 1), (1, 0))
    goal_x_max = np.float64(TCP_lims[0, 1] * 0.6).item()
    goal_x_min = 0.0#np.float64(TCP_lims[0, 0] * 0.6).item()
    goal_y_max = np.float64(TCP_lims[1, 1] * 0.6).item()
    goal_y_min = np.float64(TCP_lims[1, 0] * 0.6).item()
    goal_ranges = [goal_x_min, goal_x_max, goal_y_min, goal_y_max]

    env_modes['tcp_lims'] = TCP_lims.tolist()
    env_modes['goal_edges'] = goal_edges
    env_modes['goal_ranges'] = goal_ranges

    env = ObjectPushEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    # create controllable parameters on GUI
    action_ids = []
    min_action = env.min_action
    max_action = env.max_action

    if show_gui:
        if env_modes["movement_mode"] == "y":
            action_ids.append(
                env._pb.addUserDebugParameter("dy", min_action, max_action, 0)
            )

        if env_modes["movement_mode"] == "yRz":
            action_ids.append(
                env._pb.addUserDebugParameter("dy", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dRz", min_action, max_action, 0)
            )

        elif env_modes["movement_mode"] == "xyRz":
            action_ids.append(
                env._pb.addUserDebugParameter("dx", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dy", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dRz", min_action, max_action, 0)
            )

        elif env_modes["movement_mode"] == "TyRz":
            action_ids.append(
                env._pb.addUserDebugParameter("dTy", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dRz", min_action, max_action, 0)
            )

        elif env_modes["movement_mode"] == "TxTyRz":
            action_ids.append(
                env._pb.addUserDebugParameter("dTx", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dTy", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dRz", min_action, max_action, 0)
            )

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    # with RedirectStream(sys.stdout):
    main()
