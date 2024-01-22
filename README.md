# Tactile Pushing using Reinforcement Learning 

### Installation ###
This packages has only been developed and tested with Ubuntu 20.04 and python 3.9.

Installing tactile gym:
```
git clone https://github.com/maxyang27896/tactile_gym_push.git
cd tactile_gym_dev
pip install -e .
```

Installing mbrl-lib
```
cd tactile_gym_dev/utils/mbrl-lib
pip install -e .
```

Install PyTorch from the installation [page](https://pytorch.org/get-started/locally/).

### Training Mode-Free RL Agents ###

Run:
The pushing environment has been trained using SAC from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for all training, helper scripts are provided in `tactile_gym/sb3_helpers/`. Inside the `/sb3_helpers` directory, run
```
python train_agent.py 
```
Progress plots and saved models will be in `/sb3_helpers/saved_models/object_push-v0` folder. 

### Training Mode-Based RL Agents ###
For model-based RL, the agent has been trained using PETs from [mbrl-lib](https://github.com/facebookresearch/mbrl-lib). Some examples of usage are provided in the `/utils/mbrl-lib/notebooks folder`.

To start a training, inside the `/mbrl_helpers` directory, run
```
python object_push_training.py --num_trials 100 
```
The models and progress will be saved under `/mbrl_helpers/training_model` by default.