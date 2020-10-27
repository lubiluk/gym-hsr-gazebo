import rospy
import std_srvs.srv
import gazebo_msgs.srv

MODEL_STATE_TOPIC = '/gazebo/model_states'


class Simulator:
    def __init__(self):
        self._model_state_msg = None

        self._connect_services()
        self._connect_sub_pub()

    def _connect_services(self):
        self.pause = rospy.ServiceProxy(
            '/gazebo/pause_physics', std_srvs.srv.Empty)
        self.unpause = rospy.ServiceProxy(
            '/gazebo/unpause_physics', std_srvs.srv.Empty)
        self._get_physics_properties = rospy.ServiceProxy(
            '/gazebo/get_physics_properties', gazebo_msgs.srv.GetPhysicsProperties)
        self._set_physics_properties = rospy.ServiceProxy(
            '/gazebo/set_physics_properties', gazebo_msgs.srv.SetPhysicsProperties)
        rospy.wait_for_service('/gazebo/get_physics_properties')

    def _connect_sub_pub(self):
        self._model_state_sub = rospy.Subscriber(
            MODEL_STATE_TOPIC,
            gazebo_msgs.msg.ModelStates, self._handle_model_state)

    def go_turbo(self):
        props = self._get_physics_properties()
        props.max_update_rate = 0

        self._set_physics_properties(
            props.time_step, props.max_update_rate, props.gravity, props.ode_config)

    def _handle_model_state(self, msg):
        self._model_state_msg = msg
