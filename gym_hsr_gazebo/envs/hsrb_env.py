import gym
from gym import spaces
import numpy as np
import rospy
import roslaunch
from pathlib2 import Path
from .simulator import Simulator
from .robot import Robot
import tf
import copy


class HsrbEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    _arm_joints = [
        'arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
        'wrist_flex_joint', 'wrist_roll_joint'
    ]
    _arm_limits = [
        (0.0, 0.69),
        (-2.617, 0.0),
        (-1.919, 3.665),
        (-1.919, 1.221),
        (-1.919, 3.665),
    ]
    _base_joints = ['odom_x', 'odom_y', 'odom_t']
    _base_limits = [(-1.0, 1.0), (-1.0, 1.0), (-1.57, 1.57)]
    _launch_file = "hsrb_reach.launch"

    def __init__(self, gui=False, timestep=0.1, workspace_radius=1.0):
        super(HsrbEnv, self).__init__()
        self.gui = gui
        self.timestep = timestep
        self.workspace_radius = workspace_radius

        self._launch_world()
        rospy.init_node('gym_hsrb_env', anonymous=True)
        self._simulator = Simulator()
        self._simulator.go_turbo()
        self._robot = Robot()
        self._setup_scene()

        self.action_space = spaces.Box(-1.0,
                                       1.0,
                                       shape=(len(self._arm_joints) +
                                              len(self._base_joints), ),
                                       dtype="float32")

        self.observation_space = self._get_observation_space()

    def __del__(self):
        self.close()

    def reset(self):
        raise "Not implemented"

    def step(self, action):
        raise "Not implemented"

    def render(self, mode="human"):
        pass

    def close(self):
        self.launch.shutdown()

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def _perform_action(self, action):
        action = list(action)
        assert len(action) == len(self.action_space.high.tolist())

        arm_angles = action[:len(self._arm_joints)]
        base_positions = action[len(self._arm_joints):]

        rescaled_angles = [
            self._rescale_feature(self._arm_limits, i, f)
            for (i, f) in enumerate(arm_angles)
        ]
        rescaled_positions = [
            self._rescale_feature(self._base_limits, i, f)
            for (i, f) in enumerate(base_positions)
        ]

        self._simulator.unpause()
        self._robot.set_arm_angles(self._arm_joints, rescaled_angles)
        self._robot.set_base_position(self._base_joints, rescaled_positions)
        rospy.sleep(self.timestep)
        self._simulator.pause()

    def _setup_scene(self):
        self._model = "hsrb"
        self._table = "Lack"
        self._obj = "wood_cube_5cm"

        self._simulator.unpause()
        self._simulator.set_model_position(self._table, [1.0, 0.0, 0.0],
                                           [0, 0, 0, 1])
        self._simulator.set_model_position(self._obj, [1.0, 0.0, 0.6],
                                           [0, 0, 0, 1])
        # Let things fall down
        rospy.sleep(2.0)
        self._simulator.pause()

        self._table_start_pose = self._simulator.get_model_pose(self._table)
        self._table_away_pose = copy.deepcopy(self._table_start_pose)
        self._table_away_pose.position.x = 10
        self._table_away_pose.position.y = 10
        self._obj_start_pose = self._simulator.get_model_pose(self._obj)
        self._obj_away_pose = copy.deepcopy(self._obj_start_pose)
        self._obj_away_pose.position.x = 10
        self._obj_away_pose.position.y = 10

    def _launch_world(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        base_path = Path(__file__).parent.parent / "assets"
        launch_path = base_path / "launch" / self._launch_file

        args = ["base_dir:=" + str(base_path.resolve()), "fast_physics:=true"]

        self.launch = roslaunch.parent.ROSLaunchParent(
            uuid, [(str(launch_path.resolve()), args)], is_core=True)
        self.launch.start()

    def _reset_scene(self):
        # Get things out of the way
        self._simulator.set_model_pose(self._table, self._table_away_pose)
        self._simulator.set_model_pose(self._obj, self._obj_away_pose)

        # Reset the robot
        self._simulator.unpause()
        self._reset_robot()

        # move things to their place
        self._simulator.set_model_position(self._model, [0, 0, 0], [0, 0, 0, 1])
        self._simulator.set_model_pose(self._table, self._table_start_pose)
        obj_pos = self._sample_goal()
        rospy.sleep(2)
        self._simulator.pause()
        self._simulator.set_model_position(self._obj, obj_pos, [0, 0, 0, 1])

        return obj_pos

    def _reset_robot(self):
        self._robot.move_to_start_pose()

    def _get_joint_states(self):
        joint_p = self._robot.get_joint_states(self._arm_joints)
        scaled = [
            self._scale_feature(self._arm_limits, i, f)
            for (i, f) in enumerate(joint_p)
        ]
        return scaled

    def _get_odom(self):
        pos = self._robot.get_odom()
        quaternion = (pos.orientation.x, pos.orientation.y, pos.orientation.z,
                      pos.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)

        xyt = [pos.position.x, pos.position.y, euler[2]]
        scaled = [
            self._scale_feature(self._base_limits, i, f)
            for (i, f) in enumerate(xyt)
        ]
        return scaled

    def _get_observation_space(self):
        raise "Not implemented"

    def _get_observation(self):
        raise "Not implemented"

    def _sample_goal(self):
        return np.append((
            (np.random.sample(2) * 0.5 - 0.25) +
            [self._obj_start_pose.position.x, self._obj_start_pose.position.y
             ]).astype(np.float32), self._obj_start_pose.position.z)

    def _is_table_displaced(self):
        pose = self._simulator.get_model_pose(self._table)
        current_pose = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
                                dtype=np.float32)

        table_pose = np.array([
            self._table_start_pose.position.x,
            self._table_start_pose.position.y,
            self._table_start_pose.position.z,
            self._table_start_pose.orientation.x,
            self._table_start_pose.orientation.y,
            self._table_start_pose.orientation.z,
            self._table_start_pose.orientation.w,
        ])

        return not np.allclose(table_pose, current_pose, atol=0.01)

    def _is_robot_far_away(self):
        table_pose = self._simulator.get_model_pose(self._table)
        robot_pose = self._simulator.get_model_pose(self._model)

        return abs(table_pose.position.x - robot_pose.position.x) > self.workspace_radius or \
            abs(table_pose.position.y - robot_pose.position.y) > self.workspace_radius

    def _rescale_feature(self, lookup_table, index, value):
        r = lookup_table[index]
        return (r[1] - r[0]) * (value + 1) / 2 + r[0]

    def _scale_feature(self, lookup_table, index, value):
        r = lookup_table[index]
        return (value - r[0]) * 2 / (r[1] - r[0]) - 1
