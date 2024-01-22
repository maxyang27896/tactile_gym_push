import gym
import torch as th
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os

from stable_baselines3 import PPO, SAC
# from sb3_contrib import RAD_SAC, RAD_PPO

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.rl_utils import make_training_envs, make_eval_env
from tactile_gym.sb3_helpers.eval_agent_utils import final_evaluation
from tactile_gym.sb3_helpers.rl_plot_utils import plot_train_and_eval

# Replace with model directory
save_dir = r"/home/qt21590/Documents/Projects/tactile_gym_mbrl/tactile_gym_dev/tactile_gym/sb3_helpers/saved_models/object_push-v0/sac/s1_tactile_pose_updated"
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