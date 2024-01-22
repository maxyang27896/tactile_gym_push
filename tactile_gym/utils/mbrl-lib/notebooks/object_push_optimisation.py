from __future__ import print_function
import argparse
import copy 
import cv2
import multiprocessing as mp
import pathlib
import pickle
import time
from typing import Sequence, Tuple, cast
import os 
import functools

import gym.wrappers
import numpy as np
import omegaconf
import skvideo.io
import torch

import mbrl.planning;
import mbrl.util
from mbrl.util.env import EnvHandler
from mbrl.util.pybullet import FreezePybullet, PybulletEnvHandler
from mbrl.util.plot_and_save_push_data import plot_and_save_push_plots

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters

# produce a display to render image
from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

from contextlib import contextmanager
import sys, os

DATA_COLUMN =  [
    'trial',
    'trial_steps', 
    'time_steps', 
    'tcp_x',
    'tcp_y',
    'tcp_z',
    'contact_x', 
    'contact_y', 
    'contact_z', 
    'tcp_Rz', 
    'contact_Rz', 
    'goal_x', 
    'goal_y', 
    'goal_Rz',
    'action_y',
    'action_Rz',
    'goal_reached', 
    'rewards', 
    'contact', 
    'dones',
    ]

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

env__: gym.Env
handler__: EnvHandler

def init(env_name: str, seed: int, env_kwargs):
    global env__
    global handler__
    with suppress_stdout():
        handler__ = mbrl.util.create_handler_from_str(env_name)
        env__ = handler__.make_env_from_str(env_name, **env_kwargs)
        env__.seed(seed)
        env__.reset()

def evaluate_all_action_sequences(
    action_sequences: Sequence[Sequence[np.ndarray]],
    pool: mp.Pool,  # type: ignore
    current_state: Tuple,
) -> torch.Tensor:

    res_objs = [
        pool.apply_async(evaluate_sequence_fn, (sequence, current_state))  # type: ignore
        for sequence in action_sequences
    ]
    res = [res_obj.get() for res_obj in res_objs]
    return torch.tensor(res, dtype=torch.float32)


def evaluate_sequence_fn(action_sequence: np.ndarray, current_state: Tuple) -> float:
    global env__
    global handler__
    # obs0__ is not used (only here for compatibility with rollout_env)
    obs0 = env__.observation_space.sample()
    env = {"env": env__}
    handler__.set_env_state(current_state, env)
    _, rewards_, _ = handler__.rollout_env(
        env, obs0, -1, agent=None, plan=action_sequence
    )
    return rewards_.sum().item()


def get_random_trajectory(horizon):
    global env__
    return [env__.action_space.sample() for _ in range(horizon)]

if __name__ == "__main__":

    algo_name = 'ppo'
    env_name = 'object_push-v0'
    rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
    rl_params["env_modes"][ 'observation_mode'] = 'tactile_pose_relative_data'
    rl_params["env_modes"][ 'control_mode'] = 'TCP_position_control'
    rl_params["env_modes"][ 'use_contact'] = True
    rl_params["env_modes"][ 'traj_type'] = 'point'
    rl_params["env_modes"][ 'task'] = "goal_pos"
    rl_params["env_modes"]['planar_states'] = True
    rl_params["env_modes"]['terminate_early']  = True
    rl_params["env_modes"]['terminate_terminate_early'] = True

    rl_params["env_modes"]['rand_init_orn'] = False
    # rl_params["env_modes"]['rand_init_pos_y'] = True
    # rl_params["env_modes"]['rand_obj_mass'] = True

    rl_params["env_modes"]['additional_reward_settings'] = 'john_guide_off_normal'
    rl_params["env_modes"]['terminated_early_penalty'] =  -500
    rl_params["env_modes"]['reached_goal_reward'] = 100
    rl_params["env_modes"]['max_no_contact_steps'] = 40
    rl_params["env_modes"]['max_tcp_to_obj_orn'] = 30/180 * np.pi
    rl_params["env_modes"]['importance_obj_goal_pos'] = 1.0
    rl_params["env_modes"]['importance_obj_goal_orn'] = 1.0
    rl_params["env_modes"]['importance_tip_obj_orn'] = 1.0

    rl_params["env_modes"]['mpc_goal_orn_update'] = True
    rl_params["env_modes"]['goal_orn_update_freq'] = 'every_step'

    rl_params["env_modes"]['eval_mode'] = True
    rl_params["env_modes"]['eval_num'] = 1
    rl_params["env_modes"]['goal_list'] = [[0.1, 0.18]]

    # set limits and goals
    TCP_lims = np.zeros(shape=(6, 2))
    TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, 0.4  # x lims
    TCP_lims[1, 0], TCP_lims[1, 1] = -0.3, 0.3  # y lims
    TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
    TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
    TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
    TCP_lims[5, 0], TCP_lims[5, 1] = -180 * np.pi / 180, 180 * np.pi / 180  # yaw lims

    # goal parameter
    goal_edges = [(0, -1), (0, 1), (1, 0)] # Top bottom and stright
    # goal_edges = [(1, 0)]
    goal_x_max = np.float64(TCP_lims[0, 1] * 0.8).item()
    goal_x_min = 0.0 # np.float64(TCP_lims[0, 0] * 0.6).item()
    goal_y_max = np.float64(TCP_lims[1, 1] * 0.6).item()
    goal_y_min = np.float64(TCP_lims[1, 0] * 0.6).item()
    goal_ranges = [goal_x_min, goal_x_max, goal_y_min, goal_y_max]

    rl_params["env_modes"]['tcp_lims'] = TCP_lims.tolist()
    rl_params["env_modes"]['goal_edges'] = goal_edges
    rl_params["env_modes"]['goal_ranges'] = goal_ranges

    env_kwargs={
        'show_gui':False,
        'show_tactile':False,
        'max_steps':rl_params["max_ep_len"],
        'image_size':rl_params["image_size"],
        'env_modes':rl_params["env_modes"],
    }

    mp.set_start_method("spawn")
    handler_env_name = "pybulletgym___" + env_name
    handler = mbrl.util.create_handler_from_str(handler_env_name)
    eval_env = handler.make_env_from_str(handler_env_name, **env_kwargs)
    seed = 0
    eval_env.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    current_obs = eval_env.reset()
    render = True

    num_steps = 600
    optimizer_type = "cem"
    num_processes = 4
    samples_per_process = 256
    horizon = 25
    iterations = 4
    record_every_n_frames = 1

    if optimizer_type == "cem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.CEMOptimizer",
                "device": "cpu",
                "num_iterations": iterations,
                "elite_ratio": 0.1,
                "population_size": num_processes * samples_per_process,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
            }
        )
    elif optimizer_type == "mppi":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.MPPIOptimizer",
                "num_iterations": 5,
                "gamma": 1.0,
                "population_size": num_processes * samples_per_process,
                "sigma": 0.95,
                "beta": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "device": "cpu",
            }
        )
    elif optimizer_type == "icem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.ICEMOptimizer",
                "num_iterations": 2,
                "elite_ratio": 0.1,
                "population_size": num_processes * samples_per_process,
                "population_decay_factor": 1.25,
                "colored_noise_exponent": 2.0,
                "keep_elite_frac": 0.1,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": "true",
                "device": "cpu",
            }
        )
    else:
        raise ValueError

    controller = mbrl.planning.TrajectoryOptimizer(
            optimizer_cfg,
            eval_env.action_space.low,
            eval_env.action_space.high,
            horizon,
        )

    # Data Directory
    control_filename = 'saved_control'
    work_dir = os.path.join(os.getcwd(), control_filename, 'h_{}_iter{}_samples_{}'.format(horizon, iterations, samples_per_process))
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)


    # Record video
    if render:
        render_img = eval_env.render(mode="rgb_array")
        render_img_size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(work_dir, "optimal_policy.mp4"),
            fourcc,
            24.0,
            render_img_size,
        )
    
    with mp.Pool(
        processes=num_processes, initializer=init, initargs=[handler_env_name, seed, env_kwargs]
    ) as pool__:

        # initialise paramters
        total_reward__ = 0
        steps__ = 0
        pb_steps__ = 0.0
        done__ = False
        evaluation_result = []

        # Optimisation parameters
        max_population_size = optimizer_cfg.population_size
        if isinstance(controller.optimizer, mbrl.planning.ICEMOptimizer):
            max_population_size += controller.optimizer.keep_elite_size
        value_history = np.zeros(
            (num_steps, max_population_size, optimizer_cfg.num_iterations)
        )
        values_sizes = []  # for icem
        eval_env_dict = {"env": eval_env}
        # state_before = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()

        # Record data
        (tcp_pos_workframe, 
        tcp_rpy_workframe,
        cur_obj_pos_workframe, 
        cur_obj_rpy_workframe) = eval_env.get_obs_workframe()
        evaluation_result.append(np.hstack([0, 
                                            steps__, 
                                            pb_steps__,
                                            tcp_pos_workframe, 
                                            cur_obj_pos_workframe, 
                                            tcp_rpy_workframe[2],
                                            cur_obj_rpy_workframe[2],
                                            eval_env.goal_pos_workframe[0:2], 
                                            eval_env.goal_rpy_workframe[2],
                                            np.array([0, 0]),
                                            eval_env.goal_updated,
                                            total_reward__, 
                                            done__,
                                            False]))

        # Start control loop
        for t in range(num_steps):
            start = time.time()
            current_state__ = handler.get_current_state(
                        eval_env_dict
                    )

            def trajectory_eval_fn(action_sequences):
                return evaluate_all_action_sequences(
                    action_sequences,
                    pool__,
                    current_state__,
                )
            
            best_value = [-np.inf]  # this is hacky, sorry

            def compute_population_stats(_population, values, opt_step):
                value_history[t, : len(values), opt_step] = values.numpy()
                values_sizes.append(len(values))
                best_value[0] = max(best_value[0], values.max().item())

            plan = controller.optimize(
                trajectory_eval_fn, callback=compute_population_stats
            )
            
            # state_after = eval_env.get_tactile_pose_obs_goal_exluded()

            action__ = plan[0]
            next_obs__, reward__, done__, info__ = eval_env.step(action__)

            total_reward__ += reward__
            steps__ += 1
            pb_steps__ += info__["num_of_pb_steps"]

            if done__:
                current_goal_reached = eval_env.single_goal_reached
            else:
                current_goal_reached = eval_env.goal_updated

            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = eval_env.get_obs_workframe()
            evaluation_result.append(np.hstack([0,
                                                steps__,
                                                pb_steps__ * eval_env._sim_time_step,
                                                tcp_pos_workframe, 
                                                cur_obj_pos_workframe, 
                                                tcp_rpy_workframe[2],
                                                cur_obj_rpy_workframe[2],
                                                eval_env.goal_pos_workframe[0:2], 
                                                eval_env.goal_rpy_workframe[2],
                                                action__,
                                                current_goal_reached,
                                                total_reward__, 
                                                info__["tip_in_contact"],
                                                done__]))

            # print("goal: ", eval_env.goal_pos_workframe)
            # print("orn reward: ", eval_env.orn_obj_dist_to_goal())
            # print('reward_mode', eval_env.additional_reward_settings)

            # state_after_action_dict = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()
            # state_after_action_env = eval_env.get_tactile_pose_obs_goal_exluded()


            # Record video at every n trials
            if render and t % record_every_n_frames == 0:
                render_img = eval_env.render(mode="rgb_array")
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            print(
                f"step: {t}, time: {time.time() - start: .3f}, "
                f"reward: {reward__: .3f}, pred_value: {best_value[0]: .3f}, "
                f"total_reward: {total_reward__: .3f}"
            )

            if done__: 
                print('Episode finished...')
                break
            

         # release video at every n trials
        if render:
            out.release()

        print("total_reward: ", total_reward__)
        np.save(os.path.join(work_dir , "value_history.npy"), value_history)

        # plot and save data 
        evaluation_result = np.array(evaluation_result)
        plot_and_save_push_plots(eval_env, evaluation_result, DATA_COLUMN, 1, work_dir, "evaluation")






        ########## Test algorithm ##########
        # eval_env_dict = {"env": eval_env}
        # current_state = handler.get_current_state(
        #         eval_env_dict
        #     )
        # state_before = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()

        # action_sequences = torch.from_numpy(np.array([eval_env.action_space.sample() for _ in range(15)])).float()
        # action_sequences = torch.tile(action_sequences, (5,1,1))

        # rewards_ = evaluate_all_action_sequences(
        #     action_sequences,
        #     pool__,
        #     current_state
        # )
        # state_after = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()

        # print(rewards_)
        # print(state_before)
        # print(state_after)
        # print(state_after_action_env)
        # print(state_after_action_dict)
        ########## Test algorithm ##########