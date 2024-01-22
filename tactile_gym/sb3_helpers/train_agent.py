import gym
import os
import sys
import time
import numpy as np
import signal
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3 import PPO, SAC

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.rl_utils import make_training_envs, make_eval_env
from tactile_gym.sb3_helpers.eval_agent_utils import final_evaluation
from tactile_gym.utils.general_utils import (
    save_json_obj,
    print_sorted_dict,
    convert_json,
    check_dir,
)
from tactile_gym.sb3_helpers.custom.custom_callbacks import (
    FullPlottingCallback,
    ProgressBarManager,
)

def train_agent(
        algo_name='ppo',
        env_name='edge_follow-v0',
        rl_params={},
        algo_params={},
        augmentations=None,
    ):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # create save dir
    save_dir = os.path.join(
        "saved_models/", rl_params["env_name"], algo_name, "s{}_{}".format(rl_params["seed"], rl_params["env_modes"]["observation_mode"])
    )
    check_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # save params
    save_json_obj(convert_json(rl_params), os.path.join(save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(save_dir, "algo_params"))
    if 'rad' in algo_name:
        save_json_obj(convert_json(augmentations), os.path.join(save_dir, "augmentations"))

    # load the envs
    env = make_training_envs(
        env_name,
        rl_params,
        save_dir
    )

    rl_params["env_modes"]['eval_mode'] = True
    rl_params["env_modes"]['eval_num'] = rl_params["n_eval_episodes"]
    eval_env = make_eval_env(
        env_name,
        rl_params,
        show_gui=False,
        show_tactile=False,
    )

    # define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "trained_models/"),
        log_path=os.path.join(save_dir, "trained_models/"),
        eval_freq=rl_params["eval_freq"],
        n_eval_episodes=rl_params["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    plotting_callback = FullPlottingCallback(log_dir=save_dir, total_timesteps=rl_params['total_timesteps'])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params['eval_freq']*rl_params['n_envs'], callback=plotting_callback)

    # create the model with hyper params
    if algo_name == 'ppo':
        model = PPO(
            rl_params["policy"],
            env,
            **algo_params,
            verbose=1,
            device=device,
        )
    elif algo_name == 'sac':
        model = SAC(
            rl_params["policy"],
            env,
            **algo_params,
            verbose=1,
            device=device,
        )
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))

    # train an agent
    with ProgressBarManager(
        rl_params["total_timesteps"]
    ) as progress_bar_callback:
        model.learn(
            total_timesteps=rl_params["total_timesteps"],
            callback=[progress_bar_callback, eval_callback, event_plotting_callback],
        )

    # save the final model after training
    model.save(os.path.join(save_dir, "trained_models", "final_model"))
    env.close()
    eval_env.close()

    # run final evaluation over 20 episodes and save a vid
    final_evaluation(
        saved_model_dir=save_dir,
        n_eval_episodes=10,
        seed=None,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        render=True,
        save_vid=True,
        take_snapshot=False
    )

if __name__ == '__main__':

    # choose which RL algo to use
    algo_name = 'sac'
    env_name = 'object_push-v0'

    # import paramters
    rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
    algo_params["buffer_size"] = int(1e6) # For sac only
    algo_params["learning_rate"] = 8e-5
    algo_params["batch_size"] = 4096

    rl_params["total_timesteps"] = 5000000
    rl_params["eval_freq"] = 5000
    rl_params["n_envs"] = 10
    rl_params["max_ep_len"] = 1000

    # Define environment parameters
    rl_params["env_modes"]["observation_mode"] = "goal_aware_tactile_pose"     # tested options: tactile_and_feature, goal_aware_tactile_pose
    rl_params["env_modes"]['control_mode'] = 'TCP_position_control'
    rl_params["env_modes"]['terminate_early']  = False
    rl_params["env_modes"]['terminate_terminate_early'] = False
    rl_params["env_modes"]['use_contact'] = True

    rl_params["env_modes"]['importance_obj_goal_pos'] = 1.0
    rl_params["env_modes"]['importance_obj_goal_orn'] = 1.0
    rl_params["env_modes"]['importance_tip_obj_orn'] = 1.0 

    rl_params["env_modes"]['terminated_early_penalty'] = 0
    rl_params["env_modes"]['reached_goal_reward'] = 0

    rl_params["env_modes"]['max_no_contact_steps'] = 1000
    rl_params["env_modes"]['max_tcp_to_obj_orn'] = 180/180 * np.pi

    # set limits and goals
    TCP_lims = np.zeros(shape=(6, 2))
    TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, 0.4  # x lims
    TCP_lims[1, 0], TCP_lims[1, 1] = -0.3, 0.3  # y lims
    TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
    TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
    TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
    TCP_lims[5, 0], TCP_lims[5, 1] = -180 * np.pi / 180, 180 * np.pi / 180  # yaw lims

    # goal parameter
    goal_edges = [(0, -1), (0, 1), (1, 0)] # Top, bottom and stright
    goal_x_max = np.float64(TCP_lims[0, 1] * 0.8).item()
    goal_x_min = 0.0 
    goal_y_max = np.float64(TCP_lims[1, 1] * 0.6).item()
    goal_y_min = np.float64(TCP_lims[1, 0] * 0.6).item()
    goal_ranges = [goal_x_min, goal_x_max, goal_y_min, goal_y_max]

    rl_params["env_modes"]['tcp_lims'] = TCP_lims.tolist()
    rl_params["env_modes"]['goal_edges'] = goal_edges
    rl_params["env_modes"]['goal_ranges'] = goal_ranges

    train_agent(
        algo_name,
        env_name,
        rl_params,
        algo_params,
        augmentations
    )