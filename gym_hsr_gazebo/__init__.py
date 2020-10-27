from gym.envs.registration import register

register(
    id='HsrbPush-v0',
    entry_point='gym_hsr_gazebo.envs:HsrbPushEnv',
)