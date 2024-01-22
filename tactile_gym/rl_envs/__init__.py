from gym.envs.registration import register

register(
    id='edge_follow-v0',
    entry_point='tactile_gym.rl_envs.exploration.edge_follow.edge_follow_env:EdgeFollowEnv',
)

register(
    id='surface_follow-v0',
    entry_point='tactile_gym.rl_envs.exploration.surface_follow.surface_follow_auto.surface_follow_auto_env:SurfaceFollowAutoEnv',
)

register(
    id='surface_follow-v1',
    entry_point='tactile_gym.rl_envs.exploration.surface_follow.surface_follow_goal.surface_follow_goal_env:SurfaceFollowGoalEnv',
)

register(
    id='object_roll-v0',
    entry_point='tactile_gym.rl_envs.nonprehensile_manipulation.object_roll.object_roll_env:ObjectRollEnv',
)

register(
    id='object_push-v0',
    entry_point='tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env:ObjectPushEnv',
)

register(
    id='object_balance-v0',
    entry_point='tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.object_balance_env:ObjectBalanceEnv',
)

register(
    id='hand_manip_object-v0',
    entry_point='tactile_gym.rl_envs.prehensile_manipulation.graspers.hand_manip_object_env:HandManipObjectEnv',
)

register(
    id='hand_flip_object-v0',
    entry_point='tactile_gym.rl_envs.prehensile_manipulation.graspers.hand_flip_object_env:HandFlipObjectEnv',
)

register(
    id='hand_spin_object-v0',
    entry_point='tactile_gym.rl_envs.prehensile_manipulation.graspers.hand_spin_object_env:HandSpinObjectEnv',
)

register(
    id='hand_grasp_object-v0',
    entry_point='tactile_gym.rl_envs.prehensile_manipulation.graspers.hand_grasp_object_env:HandGraspObjectEnv',
)
