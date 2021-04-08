import gym
import gym_hsr_gazebo

env = gym.make('HsrbReach-v0')

for _ in range(10):
    env.reset()
    for _ in range(20):
        env.step([1.0] * 8)