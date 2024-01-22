from IPython import display
import argparse
import cv2
import copy
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

import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
from mbrl.util.plot_and_save_push_data import plot_and_save_training, plot_and_save_push_plots

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.utils.general_utils import check_dir

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

# Hacky evaluation for saving computation time
def evaluate_callback(env, agent, save_and_plot_flag=False, data_directory=None):
    all_rewards = []
    result = []

    # Eval goals
    goals = np.array([
        [0.0, 0.18], 
        [0.0, -0.18], 
        [0.12, 0.18],
        [0.12, -0.18],
        [0.32, 0.18],
        [0.32, -0.18],
        [0.32, 0.0]])

    for trial in range(len(goals)):
        obs = env.reset()
        env.make_goal(goals[trial])
        agent.reset()
        done = False
        trial_reward = 0.0
        steps_trial = 0
        trial_pb_steps = 0.0

        if save_and_plot_flag:
            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = env.get_obs_workframe()
            result.append(np.hstack([trial, 
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
            
            action = agent.act(obs, **{})
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            trial_reward += reward
            trial_pb_steps += info["num_of_pb_steps"]
            steps_trial += 1

            if done:
                current_goal_reached = env.single_goal_reached
            else:
                current_goal_reached = env.goal_updated,

            if save_and_plot_flag:
                (tcp_pos_workframe, 
                tcp_rpy_workframe,
                cur_obj_pos_workframe, 
                cur_obj_rpy_workframe) = env.get_obs_workframe()
                result.append(np.hstack([trial,
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
        
        all_rewards.append(trial_reward)

    # plot evaluation results
    if save_and_plot_flag:
        result = np.array(result)
        plot_and_save_push_plots(env, result, DATA_COLUMN, len(goals), data_directory, "eval_result")

    return np.mean(all_rewards)

def train_and_plot(num_trials, model_filename, eval_best, record_video):

    # Display device setting
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    if device == 'cuda:0':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    print()

    # Define model working directorys
    work_dir = os.path.join(os.getcwd(), model_filename)
    check_dir(work_dir)
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    else:
        for filename in os.listdir(work_dir):
            file_path = os.path.join(work_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Load the environment 
    algo_name = 'pets'
    env_name = 'object_push-v0'
    rl_params, _, _ = import_parameters(env_name, algo_name)

    rl_params["env_modes"][ 'observation_mode'] = 'tactile_pose_array'
    rl_params["env_modes"][ 'control_mode'] = 'TCP_position_control'
    rl_params["env_modes"]['task'] = "goal_pos"
    rl_params["env_modes"]['use_contact'] = True
    rl_params["env_modes"]['terminate_early']  = True
    rl_params["env_modes"]['terminate_terminate_early'] = True

    rl_params["env_modes"]['rand_init_orn'] = True
    # rl_params["env_modes"]['rand_init_pos_y'] = True
    # rl_params["env_modes"]['rand_obj_mass'] = True

    rl_params["env_modes"]['terminated_early_penalty'] =  -500
    rl_params["env_modes"]['reached_goal_reward'] = 100
    rl_params["env_modes"]['max_no_contact_steps'] = 40
    rl_params["env_modes"]['max_tcp_to_obj_orn'] = 30/180 * np.pi
    rl_params["env_modes"]['importance_obj_goal_pos'] = 1.0
    rl_params["env_modes"]['importance_obj_goal_orn'] = 1.0
    rl_params["env_modes"]['importance_tip_obj_orn'] = 1.0

    rl_params["env_modes"]['mpc_goal_orn_update'] = True

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
    goal_x_max = np.float64(TCP_lims[0, 1] * 0.8).item()
    goal_x_min = 0.0 
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

    # training environment
    env = gym.make(env_name, **env_kwargs)
    seed = 0
    env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    trial_length = env._max_steps
    ensemble_size = 5
    initial_buffer_size = 2000
    buffer_size = num_trials * trial_length
    target_normalised = True
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 200,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model",
            "learn_logvar_bounds": False,
            "activation_fn_cfg": {
                "_target_": "torch.nn.SiLU",
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
            "target_normalize": target_normalised,
            "dataset_size": buffer_size
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    model_env = models.ModelEnvPushing(env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)

    optimizer_cfg = {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 4,
            "elite_ratio": 0.1,
            "population_size": 350,
            "alpha": 0.1,
            "device": device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 25,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": optimizer_cfg
    })

    # Create agent 
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )

    # create buffer
    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)
    common_util.rollout_agent_trajectories(
        env,
        initial_buffer_size, # initial exploration steps
        planning.RandomAgent(env),
        {}, # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )

    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=5e-4, weight_decay=5e-5)

    # Saving config files
    config_filename = 'cfg_dict'
    config_dir = os.path.join(work_dir, config_filename)
    omegaconf.OmegaConf.save(config=cfg, f=config_dir) 
    loaded = omegaconf.OmegaConf.load(config_dir)
    assert cfg == loaded

    agent_config_filename = 'agent_cfg'
    agent_config_dir = os.path.join(work_dir, agent_config_filename)
    omegaconf.OmegaConf.save(config=agent_cfg, f=agent_config_dir) 
    loaded = omegaconf.OmegaConf.load(agent_config_dir)
    assert agent_cfg == loaded

    env_kwargs_filename = 'env_kwargs'
    env_kwargs_dir = os.path.join(work_dir, env_kwargs_filename)
    omegaconf.OmegaConf.save(config=env_kwargs, f=env_kwargs_dir) 
    loaded = omegaconf.OmegaConf.load(env_kwargs_dir)
    assert env_kwargs == loaded

    # Create eval env and agent 
    eval_env_kwargs = copy.deepcopy(env_kwargs)
    eval_env_kwargs["env_modes"]['eval_mode'] = True
    eval_env_kwargs["env_modes"]['eval_num'] = 4
    eval_env = gym.make(env_name, **eval_env_kwargs)
    eval_model_env = models.ModelEnvPushing(eval_env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)
    eval_agent = planning.create_trajectory_optim_agent_for_model(
        eval_model_env,
        agent_cfg,
        num_particles=20
    )

    ######### Main PETS loop #############
    all_train_rewards = [0]
    all_eval_rewards = [0]
    total_steps_train = [0]
    total_steps_eval = [0]
    goal_reached = [0]
    trial_push_result = []
    max_eval_reward = -np.inf

    # Save training data
    save_model_freqency = 10
    record_video_frequency = 10
    start_saving = 29
    data_columns = ['trial','trial_steps', 'time_steps', 'tcp_x','tcp_y','tcp_z','contact_x', 'contact_y', 'contact_z', 'tcp_Rz', 'contact_Rz', 'goal_x', 'goal_y', 'goal_Rz', 'rewards', 'contact', 'dones']
    training_result_directory = os.path.join(work_dir, "training_result")

    # parameters
    train_losses = [0.0]
    val_scores = [0.0]

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

    for trial in range(num_trials):
        # Reset 
        obs = env.reset()    
        agent.reset()
        done = False
        trial_reward = 0.0
        trial_pb_steps = 0.0
        steps_trial = 0
        start_trial_time = time.time()

        # Record video
        if record_video and (trial+1) % record_video_frequency == 0:
            record_every_n_frames = 3
            render_img = env.render(mode="rgb_array")
            render_img_size = (render_img.shape[1], render_img.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                os.path.join(work_dir, "training_policy_trial_{}.mp4".format((trial+1))),
                fourcc,
                24.0,
                render_img_size,
            )
        
        (tcp_pos_workframe, 
        tcp_rpy_workframe,
        cur_obj_pos_workframe, 
        cur_obj_rpy_workframe) = env.get_obs_workframe()
        trial_push_result.append(np.hstack([trial, 
                                        steps_trial, 
                                        trial_pb_steps,
                                        tcp_pos_workframe, 
                                        cur_obj_pos_workframe, 
                                        tcp_rpy_workframe[2],
                                        cur_obj_rpy_workframe[2],
                                        env.goal_pos_workframe[0:2], 
                                        env.goal_rpy_workframe[2],
                                        trial_reward, 
                                        False,
                                        done]))
        
        while not done:
            if steps_trial == 0:
                # ----------------------------- Model Training ------------------------------
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats            
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                
                model_trainer.train(
                    dataset_train, 
                    dataset_val=dataset_val, 
                    num_epochs=50, 
                    patience=50, 
                    callback=train_callback,
                    silent=True)

                # save and evaluate model at regular frequencies
                if (trial+1) % save_model_freqency  == 0 and trial >= start_saving:
                    
                    # save at frequency
                    if not eval_best:
                        model_dir = os.path.join(work_dir, 'model_trial_{}'.format(trial+1))
                        os.makedirs(model_dir, exist_ok=True)
                        dynamics_model.save(str(model_dir))

                    # save best model
                    else:
                        total_eval_reward = evaluate_callback(eval_env, eval_agent)
                        all_eval_rewards.append(total_eval_reward)
                        total_steps_eval.append(total_steps_train[-1])
                        print("Evaluation reward: ", total_eval_reward)
                        if total_eval_reward > max_eval_reward:
                            max_eval_reward = total_eval_reward
                            model_dir = os.path.join(work_dir, 'best_model')
                            os.makedirs(model_dir, exist_ok=True)
                            dynamics_model.save(str(model_dir))
                            print("Saving best model at trial {} with evaluation reward {}".format(trial+1, total_eval_reward))

                replay_buffer.save(work_dir)

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, info = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer)

            obs = next_obs
            trial_reward += reward
            trial_pb_steps += info["num_of_pb_steps"]
            steps_trial += 1

            # Save data for plotting training performance
            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = env.get_obs_workframe()
            trial_push_result.append(np.hstack([trial,
                                            steps_trial,
                                            trial_pb_steps * env._sim_time_step,
                                            tcp_pos_workframe, 
                                            cur_obj_pos_workframe, 
                                            tcp_rpy_workframe[2],
                                            cur_obj_rpy_workframe[2],
                                            env.goal_pos_workframe[0:2], 
                                            env.goal_rpy_workframe[2],
                                            trial_reward, 
                                            info["tip_in_contact"],
                                            done]))
            
            # Record video at every n trials
            if record_video and (trial+1) % record_video_frequency == 0 and steps_trial % record_every_n_frames == 0:
                render_img = env.render(mode="rgb_array")
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            if steps_trial == trial_length:
                break

        all_train_rewards.append(trial_reward)
        total_steps_train.append(steps_trial + total_steps_train[-1])
        trial_time = time.time() - start_trial_time

        # Save data to csv and plot
        trial_push_result = np.array(trial_push_result)
        plot_and_save_training(env, trial_push_result, trial, data_columns, training_result_directory)
        trial_push_result = []

        # save goal reached data during training
        if env.single_goal_reached:
            goal_reached.append(trial_reward)
        else:
            goal_reached.append(0)

        # release video at every n trials
        if record_video and (trial+1) % record_video_frequency == 0:
            out.release()

        print("Trial {}, total steps {}, rewards {}, goal reached {}, time elapsed {}".format(trial+1, steps_trial, all_train_rewards[-1], env.single_goal_reached, trial_time))

        # Save and plot training curve 
        training_result = np.stack((total_steps_train[1:], all_train_rewards[1:]), axis=-1)
        pd.DataFrame(training_result).to_csv(os.path.join(work_dir, "{}_result.csv".format("train_curve")))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(total_steps_train[1:], all_train_rewards[1:], 'bs-', total_steps_train[1:], goal_reached[1:], 'rs')
        ax.set_xlabel("Samples")
        ax.set_ylabel("Trial reward")
        fig.savefig(os.path.join(work_dir, "output_train.png"))        
        plt.close(fig)

        if (trial+1) % save_model_freqency  == 0 and trial >= start_saving:
            eval_result = np.stack((total_steps_eval[1:], all_eval_rewards[1:]), axis=-1)
            pd.DataFrame(eval_result).to_csv(os.path.join(work_dir, "{}_result.csv".format("eval_curve")))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(total_steps_eval[1:], all_eval_rewards[1:], 'bs-')
            ax.set_xlabel("Samples")
            ax.set_ylabel("Eval reward")
            fig.savefig(os.path.join(work_dir, "output_eval.png"))        
            plt.close(fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of training trials.",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="training_model",
        help="Specify the folder name to which to save the results in.",
    )

    parser.add_argument(
        "--eval_best",
        type=bool,
        default=True,
        help="Specify whether to save best model or save model at fixed frequency.",
    )

    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="Specify whether record video during training.",
    )

    args = parser.parse_args()
    train_and_plot(
        args.num_trials,
        args.model_filename,
        args.eval_best,
        args.record_video,
    )