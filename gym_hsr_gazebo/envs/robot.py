from os import times
import control_msgs.msg
import actionlib
import trajectory_msgs.msg
import rospy
import geometry_msgs.msg
import sensor_msgs.srv
import nav_msgs.msg

BASE_STATE_TOPIC = '/hsrb/omni_base_controller/state'
JOINT_STATE_TOPIC = '/hsrb/joint_states'
ODOM_TOPIC = '/hsrb/odom'
JOINTS = [
    'arm_flex_joint', 'arm_lift_joint', 'arm_roll_joint', 'wrist_flex_joint',
    'wrist_roll_joint'
]


class Robot:
    def __init__(self):
        self._joint_state_msg = None
        self._base_state_msg = None
        self._odom_msg = None
        self._joint_states = {}

        self._connect_action_clients()
        self._connect_sub_pub()

    def _connect_action_clients(self):
        self.head_trajectory_client = actionlib.SimpleActionClient(
            '/hsrb/head_trajectory_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)
        self.head_trajectory_client.wait_for_server()

        self.arm_trajectory_client = actionlib.SimpleActionClient(
            '/hsrb/arm_trajectory_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)
        self.arm_trajectory_client.wait_for_server()

        self.gripper_trajectory_client = actionlib.SimpleActionClient(
            '/hsrb/gripper_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)
        self.gripper_trajectory_client.wait_for_server()

        self.base_trajectory_client = actionlib.SimpleActionClient(
            '/hsrb/omni_base_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)
        self.base_trajectory_client.wait_for_server()

    def _connect_sub_pub(self):
        self._arm_pub = rospy.Publisher(
            '/hsrb/arm_trajectory_controller/command',
            trajectory_msgs.msg.JointTrajectory,
            queue_size=1)
        self._base_pub = rospy.Publisher('/hsrb/command_velocity',
                                         geometry_msgs.msg.Twist,
                                         queue_size=1)
        self._joint_state_sub = rospy.Subscriber(JOINT_STATE_TOPIC,
                                                 sensor_msgs.msg.JointState,
                                                 self._handle_joint_state)
        self._base_state_sub = rospy.Subscriber(
            BASE_STATE_TOPIC, control_msgs.msg.JointTrajectoryControllerState,
            self._handle_base_state)
        self._odom_sub = rospy.Subscriber(ODOM_TOPIC, nav_msgs.msg.Odometry,
                                          self._handle_odometry)

        # wait to establish connection between the controller
        while (self._arm_pub.get_num_connections() == 0
               or self._base_pub.get_num_connections() == 0):
            rospy.sleep(0.1)

    def move_head_to(self, pan, tilt, time=2.0):
        self._send_trajectory(self.head_trajectory_client,
                              ['head_pan_joint', 'head_tilt_joint'], [pan, tilt],
                              time=time,
                              wait=True)

    def move_arm_to(self, angles, time=3.0):
        assert len(angles) == 5
        self._send_trajectory(self.arm_trajectory_client, [
            'arm_flex_joint', 'arm_lift_joint', 'arm_roll_joint',
            'wrist_flex_joint', 'wrist_roll_joint'
        ],
                              angles,
                              time=time,
                              wait=True)

    def move_gripper_to(self, angle, effort, time=2.0):
        self._send_trajectory(self.gripper_trajectory_client,
                              ['hand_motor_joint'], [angle],
                              efforts=[effort],
                              time=time,
                              wait=True)

    def go_to(self, x, y, t, time=10.0):
        self._send_trajectory(self.base_trajectory_client,
                              ["odom_x", "odom_y", "odom_t"], [x, y, t],
                              time=time,
                              wait=True)

    def move_to_start_pose(self):
        self.move_head_to(0, 0)
        self.move_arm_to([
            -0.023208623448968346, -2.497211361823794e-06, 1.57, -1.57,
            -0.0008994942881326295
        ])
        self.move_gripper_to(0.0, 0.1)
        self.move_head_to(0.0, -0.9)
        self.go_to(0, 0, 0)

    def set_arm_angles(self, joints, angles, time=3.0):
        assert len(joints) == len(angles)
        assert time > 0

        self._send_trajectory(self.arm_trajectory_client,
                              joints,
                              angles,
                              time=time)

    def set_base_position(self, joints, positions, time=3.0):
        assert len(joints) == len(positions)
        assert time > 0

        self._send_trajectory(self.base_trajectory_client,
                              joints,
                              positions,
                              time=time)

    def set_desired_velocities(self, velocities, timestep=0.1):
        assert (len(velocities) == 8)

        # Arm
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = [
            'arm_flex_joint', 'arm_lift_joint', 'arm_roll_joint',
            'wrist_flex_joint', 'wrist_roll_joint'
        ]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [
            self._joint_states['arm_flex_joint'] + velocities[0] * timestep,
            self._joint_states['arm_lift_joint'] + velocities[1] * timestep,
            self._joint_states['arm_roll_joint'] + velocities[2] * timestep,
            self._joint_states['wrist_flex_joint'] + velocities[3] * timestep,
            self._joint_states['wrist_roll_joint'] + velocities[4] * timestep
        ]
        p.velocities = [0, 0, 0, 0, 0]
        p.time_from_start = rospy.Time(timestep)
        traj.points = [p]

        self._arm_pub.publish(traj)

        # Base
        tw = geometry_msgs.msg.Twist()
        tw.linear.x = velocities[5]
        tw.linear.y = velocities[6]
        tw.angular.z = velocities[7]

        self._base_pub.publish(tw)

    def _send_trajectory(self,
                         client,
                         joints,
                         positions,
                         velocities=None,
                         efforts=None,
                         time=2.0,
                         wait=False):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = joints
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = positions
        if velocities is not None:
            p.velocities = velocities
        if efforts is not None:
            p.effort = efforts
        p.time_from_start = rospy.Time(time)
        traj.points = [p]
        goal.trajectory = traj

        if client.simple_state != 2:
            client.cancel_all_goals()
        client.send_goal(goal)

        if wait:
            client.wait_for_result()

    def _handle_joint_state(self, msg):
        self._joint_state_msg = msg

        for i in range(len(msg.name)):
            self._joint_states[msg.name[i]] = msg.position[i]

    def _handle_base_state(self, msg):
        self._base_state_msg = msg

        for i in range(len(msg.joint_names)):
            self._joint_states[msg.joint_names[i]] = msg.actual.positions[i]

    def _handle_odometry(self, msg):
        self._odom_msg = msg

    def get_joint_states(self, joints):
        return [self._joint_state_msg.position[ \
                self._joint_state_msg.name.index(n)] for n in joints]

    def get_odom(self):
        return self._odom_msg.pose.pose
