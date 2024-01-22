from IPython import display
import argparse
import cv2
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, shutil
import torch
import omegaconf
import time
import torch

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from mbrl.util.plot_and_save_push_data import plot_and_save_push_plots

from tactile_gym.sb3_helpers.params import import_parameters

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

def make_evaluation_goals(env, num_trials):

        # Create evenly distributed goals along the edge
        n_point_per_side, n_random = divmod(num_trials, len(env.goal_edges))
        evaluation_goals = np.array([])
        for edge in env.goal_edges:
            # random x-axis
            goal_edges = np.zeros((n_point_per_side, 2))
            if edge[0] == 0:
                if edge[1] == -1:
                    y = env.goal_y_min
                else:
                    y = env.goal_y_max
                x = np.linspace(env.goal_x_min, env.goal_x_max, num=n_point_per_side)
            # random y axis
            else:
                if edge[0] == -1:
                    x = env.goal_x_min
                else:
                    x = env.goal_x_max
                y = np.linspace(env.goal_y_min, env.goal_y_max, num=n_point_per_side)
            goal_edges[:, 0] = x
            goal_edges[:, 1] = y
            evaluation_goals = np.hstack([
                *evaluation_goals,
                *goal_edges
            ])

        # get unique goals
        evaluation_goals = evaluation_goals.reshape(n_point_per_side*len(env.goal_edges), 2)
        evaluation_goals = np.unique(evaluation_goals,axis=0)

        # Fill in rest with random goals
        for i in range(num_trials - len(evaluation_goals)):
            evaluation_goals = np.append(evaluation_goals, [np.array(env.random_single_goal())], axis=0)
        
        return evaluation_goals


def evaluate_and_plot(model_filename, model_number, num_test_trials):

    work_dir = os.path.join(os.getcwd(), model_filename)
    if model_number:
        model_dir = os.path.join(work_dir, 'model_trial_{}'.format(model_number))
        evaluation_result_directory = os.path.join(work_dir, "evaluation_result_model_{}".format(model_number))
    else:
        model_dir = os.path.join(work_dir, 'best_model')
        evaluation_result_directory = os.path.join(work_dir, "evaluation_result_best_model")

    if not os.path.exists(evaluation_result_directory):
        os.mkdir(evaluation_result_directory)
    else:
        for filename in os.listdir(evaluation_result_directory):
            file_path = os.path.join(evaluation_result_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Load the environment 
    env_name = 'object_push-v0'
    env_kwargs_file = 'env_kwargs'
    env_kwargs_dir = os.path.join(work_dir, env_kwargs_file)
    env_kwargs = omegaconf.OmegaConf.load(env_kwargs_dir)
    env_kwargs["env_modes"]['eval_mode'] = True
    env_kwargs["env_modes"]['eval_num'] = num_test_trials
    env_kwargs["env_modes"]['goal_list'] = [[0.1, 0.18]]

    env = gym.make(env_name, **env_kwargs)
    seed = 0
    env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Get cfg and agent cfg
    config_file = 'cfg_dict'
    config_dir = os.path.join(work_dir, config_file)
    cfg = omegaconf.OmegaConf.load(config_dir)
    trial_length= cfg.overrides.trial_length

    agent_config_file = 'agent_cfg'
    agent_config_dir = os.path.join(work_dir, agent_config_file)
    agent_cfg = omegaconf.OmegaConf.load(agent_config_dir)
    agent_cfg['planning_horizon'] = 40
    # agent_cfg['optimizer_cfg']['population_size'] = 500
    # agent_cfg['optimizer_cfg']['num_iterations'] = 5

    # Re-map device
    map_location = None
    if cfg['dynamics_model']['device'] != device:
        cfg['dynamics_model']['device'] = device
        agent_cfg['optimizer_cfg']['device'] = device
        map_location = torch.device(device)
        
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape, model_dir)
    model_env = models.ModelEnvPushing(env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)
    
    # Create agent 
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )

    # Main PETS loop
    # num_test_trials = 12
    all_rewards = []
    evaluation_result = []
    goal_reached = []
    plan_time = 0.0
    train_time = 0.0
    save_vid = True
    render = True

    if hasattr(env, 'goal_edges'):
        if num_test_trials >= len(env.goal_edges):
            evaluate_goals = make_evaluation_goals(env, num_test_trials)

    if save_vid:
        record_every_n_frames = 3
        render_img = env.render(mode="rgb_array")
        render_img_size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(evaluation_result_directory, "evaluated_policy.mp4"),
            fourcc,
            24.0,
            render_img_size,
        )

    evaluate_time = time.time()
    for trial in range(num_test_trials):
        obs = env.reset()  
        # env.make_goal([0.1, 0.18])
        # if hasattr(env, 'goal_edges'):
        #     if num_test_trials >= len(env.goal_edges):
        #         env.make_goal(evaluate_goals[trial])
        agent.reset()
        
        done = False
        trial_reward = 0.0
        trial_pb_steps = 0.0
        steps_trial = 0
        start_trial_time = time.time()

        (tcp_pos_workframe, 
        tcp_rpy_workframe,
        cur_obj_pos_workframe, 
        cur_obj_rpy_workframe) = env.get_obs_workframe()
        evaluation_result.append(np.hstack([trial, 
                                            steps_trial, 
                                            trial_pb_steps,
                                            tcp_pos_workframe, 
                                            cur_obj_pos_workframe, 
                                            tcp_rpy_workframe[2],
                                            cur_obj_rpy_workframe[2],
                                            env.goal_pos_workframe[0:2], 
                                            env.goal_rpy_workframe[2],
                                            np.array([0, 0]),
                                            env.goal_updated,
                                            trial_reward, 
                                            False,
                                            done]))

        while not done:

            # --- Doing env step using the agent and adding to model dataset ---
            start_plan_time = time.time()
            action = agent.act(obs, **{})
            next_obs, reward, done, info = env.step(action)
            plan_time = time.time() - start_plan_time

            if render:
                render_img = env.render(mode="rgb_array")
            else:
                render_img = None
            
            obs = next_obs
            trial_reward += reward
            trial_pb_steps += info["num_of_pb_steps"]
            steps_trial += 1

            if done:
                current_goal_reached = env.single_goal_reached
            else:
                current_goal_reached = env.goal_updated,
            
            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = env.get_obs_workframe()
            evaluation_result.append(np.hstack([trial,
                                                steps_trial,
                                                trial_pb_steps * env._sim_time_step,
                                                tcp_pos_workframe, 
                                                cur_obj_pos_workframe, 
                                                tcp_rpy_workframe[2],
                                                cur_obj_rpy_workframe[2],
                                                env.goal_pos_workframe[0:2], 
                                                env.goal_rpy_workframe[2],
                                                action,
                                                current_goal_reached,
                                                trial_reward, 
                                                info["tip_in_contact"],
                                                done]))

            # use record_every_n_frames to reduce size sometimes
            if save_vid and steps_trial % record_every_n_frames == 0:

                # warning to enable rendering
                if render_img is None:
                    sys.exit('Must be rendering to save video')

                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            if steps_trial == trial_length:
                break
        
        print("Terminated at step {} with reward {}, goal reached: {}, time elapsed {}".format(
            steps_trial, 
            trial_reward, 
            env.single_goal_reached,
            time.time() - start_trial_time)
            )
        all_rewards.append(trial_reward)

        # save goal reached data during training
        if env.single_goal_reached:
            goal_reached.append(trial_reward)
        else:
            goal_reached.append(0)

    if save_vid:
        out.release()

    print("The average reward over {} episodes is {}, time elapsed {}".format(
        num_test_trials, 
        np.mean(all_rewards),
        time.time() - evaluate_time)   
        )   

    # Save data 
    evaluation_result = np.array(evaluation_result)

    # plot evaluation results
    plot_and_save_push_plots(env, evaluation_result, DATA_COLUMN, num_test_trials, evaluation_result_directory, "evaluation")

    # Plot evaluation results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_rewards, 'bs-', goal_reached, 'rs')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Trial reward")
    fig.savefig(os.path.join(evaluation_result_directory, "evaluation_output.png"))
    plt.close(fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_filename",
        type=str,
        default="training_model",
        help="Specify the folder name to which to evaluate the results in",
    )
    parser.add_argument(
        "--model_num",
        type=int,
        default=0,
        help="Model number to test for evaluation.",
    )
    parser.add_argument(
        "--eval_trials",
        type=int,
        default=12,
        help="Number of evaluation trials.",
    )
    args = parser.parse_args()
    evaluate_and_plot(
        args.model_filename,
        args.model_num,
        args.eval_trials,
    )