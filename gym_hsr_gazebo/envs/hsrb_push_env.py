import gym
from gym import error, spaces, utils
from gym.utils import seeding
import roslaunch
import rospy
from pathlib2 import Path
import sensor_msgs.msg
import trajectory_msgs.msg
import geometry_msgs.msg
import gazebo_msgs.msg
import control_msgs.msg
import nav_msgs.msg
from simulator import Simulator
from robot import Robot
import numpy as np

TIME_STEP = 0.1
DISTANCE_THRESHOLD = 0.04


class HsrbPushEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._launch_world()
        rospy.init_node('hsrb_push_env', anonymous=True)

        self._simulator = Simulator()
        self._simulator.go_turbo()
        self._robot = Robot()


        # 5 for arm, 3 for base
        self._goal_cube_xy = self._sample_goal()
        obs = self._get_observation()
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        rospy.loginfo("Environment ready")

    def _launch_world(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        base_path = Path(__file__).parent.parent / "assets"
        launch_path = base_path / "launch" / "hsrb_push.launch"

        args = ["base_dir:="+str(base_path.resolve())]

        self.launch = roslaunch.parent.ROSLaunchParent(
            uuid, [(str(launch_path.resolve()), args)], is_core=True)
        self.launch.start()

    def reset(self):
        self._simulator.unpause()
        self._robot.move_to_start_pose()
        rospy.sleep(TIME_STEP)
        self._simulator.pause()

        self._goal_cube_xy = self._sample_goal()

        return self._get_observation()

    def step(self, action):
        self._simulator.unpause()
        self._robot.set_desired_velocities(action, TIME_STEP)
        rospy.sleep(TIME_STEP)
        self._simulator.pause()

        obs = self._get_observation()

        is_success = self._is_success(obs['achieved_goal'], self._goal_cube_xy)
        info = {
            'is_success': is_success,
        }
        reward = self.compute_reward(obs['achieved_goal'], self._goal_cube_xy, info)
        done = is_success

        return (obs, reward, done, info)

    def render(self, mode='human'):
        pass

    def close(self):
        self.launch.shutdown()

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.allclose(desired_goal, achieved_goal, atol=DISTANCE_THRESHOLD):
            return 1
        else:
            return 0

    def _get_observation(self):
        cube_pose = self._simulator.get_model_pose('wood_cube_5cm')
        cube_xy = np.array([cube_pose.position.x,
                            cube_pose.position.y], dtype=np.float32)
        joint_states = self._robot.get_joint_states()
        joint_positions = np.array(joint_states, dtype=np.float32)
        odom = self._robot.get_odom()
        odom_pose = np.array([odom.x,
                              odom.y,
                              odom.z], dtype=np.float32)

        return {
            'observation': np.concatenate([cube_xy, joint_positions, odom_pose]),
            'achieved_goal': cube_xy,
            'desired_goal': self._goal_cube_xy
        }

    def _is_success(self, achieved_goal, desired_goal):
        return np.allclose(desired_goal, achieved_goal, atol=DISTANCE_THRESHOLD)

    def _sample_goal(self):
        return (np.random.sample(2) * [0.5 + 0.5, 1]).astype(np.float32)
