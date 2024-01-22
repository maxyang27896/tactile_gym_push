import sys

def import_parameters(env_name, algo_name):

    if env_name == "object_push-v0":
        from tactile_gym.sb3_helpers.params.object_push_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            rl_params_sac_her,
            sac_params,
            rl_params_pets,
            augmentations,
        )

    else:
        sys.exit("Incorrect environment specified: {}.".format(env_name))

    if 'ppo' in algo_name:
        return rl_params_ppo, ppo_params, augmentations
    elif 'sac' in algo_name:
        if 'her' in algo_name:
            return rl_params_sac_her, sac_params, augmentations
        else:
            return rl_params_sac, sac_params, augmentations
    elif 'pets' in algo_name:
        return rl_params_pets, None, None
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))
