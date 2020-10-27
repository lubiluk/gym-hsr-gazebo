import gym
import gym_hsr_gazebo
import sys
import signal

env = gym.make('HsrbPush-v0')

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    env.close
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

env.reset()

for _ in range(10000):
    env.step([1.0] * 8)