from os import times
from .hsrb_env import HsrbEnv
from gym import spaces
import numpy as np


class HsrbReachEnv(HsrbEnv):
    def __init__(self, gui=True, timestep=0.1, workspace_radius=1.0, dense=False):
        super(HsrbReachEnv, self).__init__(gui, timestep, workspace_radius)
        self._dense = dense

    def reset(self):
        self._reset_scene()

        return self._get_observation()

    def step(self, action):
        """
        Action in terms of desired joint positions.
        The last three numbers are odom positions.
        """
        self._perform_action(action)

        obs = self._get_observation()

        is_success = self._is_success()
        is_safety_violated = self._is_table_displaced() or self._is_robot_far_away()

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated,
        }
        reward = self._compute_reward(is_success, is_safety_violated)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _compute_reward(self, is_success, is_safety_violated):
        if is_success:
            return 1.0

        if is_safety_violated:
            return -1.0

        if self._dense:
            return -0.01

        return 0.0

    def _is_success(self):
        return False

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")

    def _get_observation(self):
        goal_pos = self._simulator.get_model_pose(self._obj)
        goal_xyz = np.array([
            goal_pos.position.x, goal_pos.position.y,
            goal_pos.position.z
        ])

        joint_states = self._get_joint_states()
        joint_states = np.array(joint_states, dtype=np.float32)

        robot_xyt = self._get_odom()
        robot_xyt = np.array(robot_xyt,
                             dtype=np.float32)

        return np.concatenate([joint_states, robot_xyt,
                               goal_xyz]).astype(np.float32)

