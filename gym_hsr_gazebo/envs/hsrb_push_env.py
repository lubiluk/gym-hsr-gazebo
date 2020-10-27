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

TIME_STEP = 0.1


class HsrbPushEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._launch_world()
        rospy.init_node('hsrb_push_env', anonymous=True)

        self._simulator = Simulator()
        self._simulator.go_turbo()
        self._robot = Robot()

        rospy.loginfo("Environment ready")

    def _launch_world(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        base_path = Path(__file__).parent.parent / "assets"
        launch_path = base_path / "launch" / "hsrb_push.launch"

        args = ["base_dir:="+str(base_path.resolve())]

        self.launch = roslaunch.parent.ROSLaunchParent(
            uuid, [(str(launch_path.resolve()), args)])
        self.launch.start()

    def step(self, action):
        self._simulator.unpause()
        self._robot.set_desired_velocities(action, TIME_STEP)
        rospy.sleep(TIME_STEP)
        self._simulator.pause()

    def reset(self):
        self._simulator.unpause()
        self._robot.move_to_start_pose()
        self._simulator.pause()

    def render(self, mode='human'):
        pass

    def close(self):
        self.launch.shutdown()
