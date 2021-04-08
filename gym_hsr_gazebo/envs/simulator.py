import rospy
import std_srvs.srv
import gazebo_msgs.srv
import geometry_msgs.msg

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
        self.reset = rospy.ServiceProxy(
            '/gazebo/reset_world', std_srvs.srv.Empty)
        self._set_model_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', gazebo_msgs.srv.SetModelState)
        self._get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', gazebo_msgs.srv.GetModelState)
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

    def get_model_pose(self, model):
        if self._model_state_msg is None:
            return None
        idx = self._model_state_msg.name.index(model)
        return self._model_state_msg.pose[idx]

    def set_model_pose(self, model_name, pose):
        state = gazebo_msgs.msg.ModelState()
        state.pose =  pose
        state.reference_frame = "world"
        state.model_name = model_name

        self._set_model_state(state)

    def set_model_position(self, model_name, position, orientation):
        state = gazebo_msgs.msg.ModelState()
        state.pose.position.x = position[0]
        state.pose.position.y = position[1]
        state.pose.position.z = position[2]
        state.pose.orientation = geometry_msgs.msg.Quaternion(
            orientation[0], orientation[1], orientation[2], orientation[3])
        state.reference_frame = "world"
        state.model_name = model_name

        self._set_model_state(state)

    def get_relative_pose(self, model_name, reference_name):
        state = gazebo_msgs.msg.GetModelState()
        state.model_name = model_name
        state.relative_entity_name = reference_name

        self._get_model_state(state)
