import gym
import gym_hsr_gazebo

env = gym.make('HsrbReach-v0')

env.reset()

for _ in range(10000):
    env.step([1.0] * 8)