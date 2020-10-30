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
    'arm_flex_joint',
    'arm_lift_joint',
    'arm_roll_joint',
    'wrist_flex_joint',
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
        self.trajectory_client = actionlib.SimpleActionClient(
            '/hsrb/head_trajectory_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)
        self.trajectory_client.wait_for_server()

    def _connect_sub_pub(self):
        self._arm_pub = rospy.Publisher(
            '/hsrb/arm_trajectory_controller/command',
            trajectory_msgs.msg.JointTrajectory, queue_size=1)
        self._base_pub = rospy.Publisher(
            '/hsrb/command_velocity',
            geometry_msgs.msg.Twist, queue_size=1)
        self._joint_state_sub = rospy.Subscriber(
            JOINT_STATE_TOPIC,
            sensor_msgs.msg.JointState, self._handle_joint_state)
        self._base_state_sub = rospy.Subscriber(
            BASE_STATE_TOPIC,
            control_msgs.msg.JointTrajectoryControllerState, self._handle_base_state)
        self._odom_sub = rospy.Subscriber(
            ODOM_TOPIC,
            nav_msgs.msg.Odometry, self._handle_odometry
        )

        # wait to establish connection between the controller
        while (self._arm_pub.get_num_connections() == 0
                or self._base_pub.get_num_connections() == 0):
            rospy.sleep(0.1)

    def move_head_to(self, pan, tilt, time=2.0):
        self._send_trajectory(
            ['head_pan_joint', 'head_tilt_joint'],
            [0, 0],
            time=time
        )

    def move_arm_to(self, angles, time=3.0):
        assert len(angles) == 5
        self._send_trajectory(
            ['arm_flex_joint',
             'arm_lift_joint',
             'arm_roll_joint',
             'wrist_flex_joint',
             'wrist_roll_joint'],
            angles,
            time=time
        )

    def move_gripper_to(self, angle, effort, time=2.0):
        self._send_trajectory(
            ['hand_motor_joint'],
            [angle],
            efforts=[effort],
            time=time
        )

    def move_to_start_pose(self):
        self.move_head_to(0, 0)
        self.move_arm_to([
            -0.023208623448968346,
            -2.497211361823794e-06,
            1.57,
            -1.57,
            -0.0008994942881326295])
        self.move_gripper_to(0.0, 0.1)
        self.move_head_to(0.0, -0.9)

    def set_desired_velocities(self, velocities, timestep=0.1):
        assert(len(velocities) == 8)

        # Arm
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = [
            'arm_flex_joint',
            'arm_lift_joint',
            'arm_roll_joint',
            'wrist_flex_joint',
            'wrist_roll_joint'
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
        tw.linear.z = velocities[7]

        self._base_pub.publish(tw)

    def _send_trajectory(self, joints, positions, velocities=None, efforts=None, time=2.0):
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

        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()

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

    def get_joint_states(self):
        return [self._joint_state_msg.position[ \
                self._joint_state_msg.name.index(n)] for n in JOINTS]

    def get_odom(self):
        return self._odom_msg.pose.pose.position
