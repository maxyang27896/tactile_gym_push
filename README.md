# Tactile-Gym: RL suite for tactile robotics
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package provides a suite of [PyBullet](https://github.com/bulletphysics/bullet3) reinforcement learning environments targeted towards using tactile data as the main form of observation.

<p align="center">
  <img width="256" src="docs/readme_videos/edge_follow.gif">
  <img width="256" src="docs/readme_videos/surface_follow.gif"> <br>
  <img width="256" src="docs/readme_videos/object_roll.gif">
  <img width="256" src="docs/readme_videos/object_push.gif">
  <img width="256" src="docs/readme_videos/object_balance.gif">
</p>

- [Installation](#installation)
- [Testing Environments](#testing-environments)
- [Training Agents](#training-agents)
- [Pretrained Agents](#pretrained-agents)
- [Environment Details](#environment-details)
- [Observation Details](#observation-details)
- [Alternate Robot Arms](#preliminary-support-for-alternate-robot-arms)
- [Additional Info](#additional-info)


### Installation ###
This packages has only been developed and tested with Ubuntu 18.04 and python 3.8.

```
# TODO: install via pypi
git clone https://github.com/ac-93/tactile_gym.git
cd tactile_gym
python setup.py install
```

### Testing Environments ###

Demonstration files are provided for all environments in the example directory. For example, from the base directory run
```
python examples/demo_example_env.py
```
to run a user controllable example environment.

### Training Agents ###

The environments use the [OpenAI Gym](https://gym.openai.com/) interface so should be compatible with most reinforcement learning librarys.

We use [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for all training, helper scripts are provided in `tactile_gym/sb3_helpers/`

A simple experiment can be run with `simple_sb3_example.py`, a full training script can be run with `train_agent.py`. Experiment hyper-params are in the `parameters` directory.

**Training with image augmentations:** If intending to use image augmentations for training, as done in the paper, then [this](https://github.com/ac-93/stable-baselines3-contrib) fork of sb3 contrib is required. (**TODO: Contribute this to sb3_contrib**).

### Pretrained Agents ###

Example PPO/RAD_PPO agents, trained via SB3 are provided for all environments and all observation spaces. These can be downloaded [here](https://drive.google.com/drive/folders/1stIhPc0HBN8fcJfMq6e-wHcsp6VpJafQ?usp=sharing)
and placed in `tactile_gym/examples/enjoy`.

In order to demonstrate a pretrained agent from the base directory run
```
python examples/demo_trained_agent.py -env='env_name' -obs='obs_type' -algo='algo_name'
```

### Environment Details ###

| Environment Name | Description |
| :---: | :--- |
| `edge_follow-v0` | <ul><li>A flat edge is randomly orientated through 360 degrees and placed within the environment. </li><li>The sensor is initialised to contact a random level of pentration at the start of the edge.</li><li>The objective is to traverse the edge to a goal at the oposing end whilst maintaining that the edge is located centrally on the sensor.</li></ul>  |
| `surface_follow-v0`   | <ul><li>A terrain like surface is generated through [OpenSimplex Noise](https://pypi.org/project/opensimplex/).</li><li>The sensor is initialised in the center, touching the surface.</li><li>A goal is randomly placed towards the edges of the surface.</li><li>The objective is to maintain a normal orientation to the surface and a set penetration distance whilst the sensor is automatically moved towards the goal.</li></ul> |
| `surface_follow-v1`   | <ul><li>Same as `-v0` however the goal location is included in the observation and the agent must additionally learn to traverse towards the goal.</li></ul> |
| `object_roll-v0`   | <ul><li>A small spherical object of random size is placed on the table.</li><li>A flat tactile sensor is initialised to touch the object at a random location relative to the sensor.</li><li>A goal location is generated in the sensor frame.</li><li>The objective is to manipulate the object to the goal location.</li></ul> |
| `object_push-v0`   | <ul><li>A cube object is placed on the table and the sensor is initialised to touch the object (in a right-angle configuration).</li><li>A trajectory of points is generated through OpenSimplex Noise.</li><li>The objective is to push the object along the trajectory, when the current target point has been reached it is incremented along the trajectory until no points are left.</li></ul> |
| `object_balance-v0`   | <ul><li>Similar to a 2d CartPole environment.</li><li>An unstable pole object is balanced on the tip of a sensor pointing upwards.</li><li>A random force pertubation is applied to the object to cause instability.</li><li>The objective is to learn planar actions to counteract the rotation of the object and mantain its balanced position.</li></ul> |

### Observation Details ###

All environments contain 4 main modes of observation:

| Observation Type | Description |
| :---: | :--- |
| `oracle` | Comprises ideal state information from the simulator, which is difficult information to collect in the real world, we use this to give baseline performance for a task. The information in this state varies between environments but commonly includes TCP pose, TCP velocity, goal locations and the current state of the environment. This observation requires signifcantly less compute both to generate data and for training agent networks.|
| `tactile` | Comprises images (default 128x128) retrieved from the simulated optical tactile sensor attached to the end effector of the robot arm (Env Figures right). Where tactile information alone is not sufficient to solve a task, this observation can be extended with oracle information retrieved from the simulator. This should only include information that could be be easily and accurately captured in the real world, such as the TCP pose that is available on industrial robotic arms and the goal pose. |
| `visual` | Comprises RGB images (default 128x128) retrieved from a static, simulated camera viewing the environment (Env Figures left). Currently, only a single camera is used, although this could be extended to multiple cameras. |
| `visuotactile` |  Combines the RGB visual and tactile image observations to into a 4-channel RGBT image. This case demonstrates a simple method of multi-modal sensing. |

When additional information is required to solve a task, such as goal locations, appending `_and_feature` to the observation name will return the complete observation.

### Preliminary Support for Alternate Robot Arms ###

The majority of testing is done on the simulated UR5 robot arm. The Franka Emika Panda and Kuka LBR iiwa robot arms are additionally provided however there may be bugs when using these arms. Particularly, workframes may need to be adjusted to ensure that arms can comfortably reach all the neccessary configurations. These arms can be used by changing the `self.arm_type` flag within the code.

<p align="center">
  <img src="docs/readme_videos/surf_arm_transfer.gif">
</p>



### Additional Info ###
| Site | Arxiv | Cite |
| :---: | :---:  | :---: |
| [Tactile Gym](https://sites.google.com/view/tactile-gym) |   Arxiv Link  |  <pre>article = {<br>temp<br>temp<br>} </pre> |
