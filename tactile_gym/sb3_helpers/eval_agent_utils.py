import gym
import os, sys
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import shutil

import pybullet as pb
import stable_baselines3 as sb3

import warnings
# from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback, EventCallback

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.rl_utils import make_eval_env
from tactile_gym.utils.general_utils import load_json_obj
from tactile_gym.utils.plot_and_save_push_data import plot_and_save_push_plots

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
    'dones']


def eval_and_save_vid(
    model, env, saved_model_dir, n_eval_episodes=10, deterministic=True, render=False, save_vid=False, take_snapshot=False
):

    evaluation_result_directory = os.path.join(saved_model_dir, "evaluation_result")
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

    if save_vid:
        record_every_n_frames = 1
        render_img = env.render(mode="rgb_array")
        render_img_size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(evaluation_result_directory, "evaluated_policy.mp4"),
            fourcc,
            24.0,
            render_img_size,
        )

    if take_snapshot:
        render_img = env.render(mode="rgb_array")
        render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(saved_model_dir, "env_snapshot.png"), render_img)

    episode_rewards, episode_lengths = [], []
    evaluation_result = []
    goal_reached = []
    for episode in range(n_eval_episodes):
        obs = env.reset()
        if hasattr(env, 'goal_edges'):
            if n_eval_episodes >= len(env.goal_edges):
                env.make_goal(env.eval_goals[episode])

        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        trial_pb_steps = 0

        (tcp_pos_workframe, 
        tcp_rpy_workframe,
        cur_obj_pos_workframe, 
        cur_obj_rpy_workframe) = env.get_obs_workframe()
        evaluation_result.append(np.hstack([
            episode, 
            episode_length, 
            trial_pb_steps,
            tcp_pos_workframe, 
            cur_obj_pos_workframe, 
            tcp_rpy_workframe[2],
            cur_obj_rpy_workframe[2],
            env.goal_pos_workframe[0:2], 
            env.goal_rpy_workframe[2],
            np.array([0, 0]),
            env.goal_updated,
            episode_reward, 
            False,
            done]))

        while not done:

            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)

            episode_reward += reward
            episode_length += 1
            trial_pb_steps += _info["num_of_pb_steps"]

            # Save data to dataframe and plot
            if done:
                current_goal_reached = env.single_goal_reached
            else:
                current_goal_reached = env.goal_updated,

            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = env.get_obs_workframe()
            evaluation_result.append(np.hstack([
                episode, 
                episode_length, 
                trial_pb_steps*env._sim_time_step,
                tcp_pos_workframe, 
                cur_obj_pos_workframe, 
                tcp_rpy_workframe[2],
                cur_obj_rpy_workframe[2],
                env.goal_pos_workframe[0:2], 
                env.goal_rpy_workframe[2],
                action,
                current_goal_reached,
                episode_reward, 
                _info["tip_in_contact"],
                done]))

            # render visual + tactile observation
            if render:
                render_img = env.render(mode="rgb_array")
            else:
                render_img = None

            # write rendered image to mp4
            # use record_every_n_frames to reduce size sometimes
            if save_vid and episode_length % record_every_n_frames == 0:

                # warning to enable rendering
                if render_img is None:
                    sys.exit('Must be rendering to save video')

                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            if take_snapshot:
                render_img = env.render(mode="rgb_array")
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(saved_model_dir, "env_snapshot.png"), render_img)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # save goal reached data during training
        if env.single_goal_reached:
            goal_reached.append(episode_reward)
        else:
            goal_reached.append(0)

     # Save and plot data 
    evaluation_result = np.array(evaluation_result)
    plot_and_save_push_plots(env, evaluation_result, DATA_COLUMN, n_eval_episodes, evaluation_result_directory, "evaluation")

    if save_vid:
        out.release()

     # Plot evaluation results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episode_rewards, 'bs-', goal_reached, 'rs')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Trial reward")
    fig.savefig(os.path.join(evaluation_result_directory, "evaluation_output.png"))
    plt.close(fig)

    return episode_rewards, episode_lengths


def final_evaluation(
    saved_model_dir,
    n_eval_episodes,
    seed=None,
    deterministic=True,
    show_gui=True,
    show_tactile=True,
    render=False,
    save_vid=False,
    take_snapshot=False
):

    rl_params = load_json_obj(os.path.join(saved_model_dir, "rl_params"))
    algo_params = load_json_obj(os.path.join(saved_model_dir, "algo_params"))
    
    rl_params["env_modes"]['eval_mode'] = True
    rl_params["env_modes"]['eval_num'] = rl_params["n_eval_episodes"]
    # create the evaluation env
    # eval_env = make_eval_env(
    #     rl_params["env_name"],
    #     rl_params,
    #     show_gui=show_gui,
    #     show_tactile=show_tactile,
    # )
    eval_env = gym.make(rl_params["env_name"],
        max_steps=rl_params["max_ep_len"],
        image_size=rl_params["image_size"],
        env_modes=rl_params["env_modes"],
        show_gui=show_gui,
        show_tactile=show_tactile,
    )

    # load the trained model
    model_path = os.path.join(saved_model_dir, "trained_models", "best_model.zip")
    # model_path = os.path.join(saved_model_dir, "trained_models", "final_model.zip")
    # replay_buffer_path = os.path.join(saved_model_dir, "trained_models", "replay_buffer")

    # create the model with hyper params
    if rl_params["algo_name"] == 'ppo':
        model = sb3.PPO.load(model_path)
    elif rl_params["algo_name"] == 'sac':
        model = sb3.SAC.load(model_path)
    elif rl_params["algo_name"] == 'sac_her':
        model = sb3.SAC.load(model_path, env=eval_env)
        # model.load_replay_buffer(replay_buffer_path)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))

    # seed the env
    if seed is not None:
        eval_env.reset()
        eval_env.seed(seed)

    # evaluate the trained agent
    episode_rewards, episode_lengths = eval_and_save_vid(
        model,
        eval_env,
        saved_model_dir,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        save_vid=save_vid,
        render=render,
        take_snapshot=take_snapshot
    )

    print(
        "Avg Ep Rew: {}, Avg Ep Len: {}".format(
            np.mean(episode_rewards), np.mean(episode_lengths)
        )
    )

    eval_env.close()

    return episode_rewards, episode_lengths 


def evaluation_callback(
    model,
    eval_env,
    n_eval_episodes,
    seed=None,
    deterministic=True,
    show_gui=True,
    show_tactile=True,
    render=False,
    save_vid=False,
    take_snapshot=False
):

    # seed the env
    if seed is not None:
        eval_env.reset()
        eval_env.seed(seed)

    # evaluate the trained agent
    episode_rewards, episode_lengths = eval_and_save_vid(
        model,
        eval_env,
        saved_model_dir,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        save_vid=save_vid,
        render=render,
        take_snapshot=take_snapshot
    )

    print(
        "Avg Ep Rew: {}, Avg Ep Len: {}".format(
            np.mean(episode_rewards), np.mean(episode_lengths)
        )
    )

    eval_env.close()

    return episode_rewards, episode_lengths 

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            # episode_rewards, episode_lengths = evaluate_policy(
            #     self.model,
            #     self.eval_env,
            #     n_eval_episodes=self.n_eval_episodes,
            #     render=self.render,
            #     deterministic=self.deterministic,
            #     return_episode_rewards=True,
            #     warn=self.warn,
            #     callback=self._log_success_callback,
            # )

            episode_rewards, episode_lengths = evaluation_callback(
                self.model,
                self.eval_env,
                self.n_eval_episodes,
                deterministic=self.deterministic,
                show_gui=False,
                show_tactile=False,
                render=False,
                save_vid=False,
                take_snapshot=False
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


if __name__ == "__main__":

    # evaluate params
    n_eval_episodes = 10
    seed = int(1)
    deterministic = True
    show_gui = True
    show_tactile = False
    render = False
    save_vid = False
    take_snapshot = False

    ## load the trained model
    # algo_name = 'ppo'
    # algo_name = 'rad_ppo'
    algo_name = 'sac'
    # algo_name = 'rad_sac'

    env_name = 'edge_follow-v0'
    # env_name = 'surface_follow-v0'
    # env_name = 'surface_follow-v1'
    # env_name = 'object_roll-v0'
    # env_name = 'object_push-v0'
    # env_name = 'object_balance-v0'

    obs_type = 'oracle'
    # obs_type = 'tactile'
    # obs_type = 'visual'
    # obs_type = 'visuotactile'

    ## combine args
    saved_model_dir = os.path.join(os.path.dirname(__file__), "saved_models","enjoy", env_name, algo_name, obs_type)

    # overwrite for testing
    # TODO: remove this
    # saved_model_dir = os.path.join("saved_models", 'edge_follow-v0', 'sac', 's1_oracle')
    # saved_model_dir = os.path.join("saved_models", 'hand_manip_object-v0', 'ppo', 's1_oracle')
    # saved_model_dir = os.path.join("saved_models", 'hand_flip_object-v0', 'ppo', 's1_oracle')
    saved_model_dir = os.path.join("saved_models", 'hand_spin_object-v0', 'ppo', 's1_oracle')
    # saved_model_dir = os.path.join("saved_models", 'hand_grasp_object-v0', 'ppo', 's1_oracle')


    final_evaluation(
        saved_model_dir,
        n_eval_episodes,
        seed=seed,
        deterministic=deterministic,
        show_gui=show_gui,
        show_tactile=show_tactile,
        render=render,
        save_vid=save_vid,
        take_snapshot=take_snapshot
    )