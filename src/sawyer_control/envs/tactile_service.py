import rospy
from wsg_50_common.msg import fingerDSAdata
from intera_core_msgs.msg import EndpointState
from sawyer_control.srv import tactile
import collections
#from geometry_msgs.msg import Wrench
import numpy as np

#data_left_buffer = collections.deque(maxlen=1)
data_left_buffer = []
force_buffer = collections.deque(maxlen=10)

def left_finger_callback(msg):
	global data_left_buffer
	#data_left_buffer.append(msg.data_array)
	data_left_buffer = msg.data_array

def endpoint_callback(msg):
	global force_buffer
	force_buffer.append(np.array((msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)))

def handle_request(req):
	print("Getting obs")
	print("force_buffer_size:%d"%(len(force_buffer)))
	data_force = np.array([force_buffer[i] for i in range(10)])
	print(data_force)
	#data_left = np.array([data_left_buffer[i] for i in range(10)])
	#data_force = np.mean(data_force, axis=0)
	data_force = np.median(data_force, axis=0)
	#data_left = np.mean(data_left, axis=0)
	data_left = np.array(data_left_buffer)
	return (data_force, data_left)

def tactile_server():
	rospy.init_node('tactile_server')
	s = rospy.Service('tactile_service', tactile, handle_request)
	print("Tactile Server Active")
	rospy.spin()

if __name__ == '__main__':
	rospy.Subscriber('/wsg_50_driver/left_finger_dsa_array', fingerDSAdata, left_finger_callback)
	rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState, endpoint_callback)
	tactile_server()