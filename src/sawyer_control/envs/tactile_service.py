import rospy
from wsg_50_common.msg import fingerDSAdata
from intera_core_msgs.msg import EndpointState
from sawyer_control.srv import tactile
#from geometry_msgs.msg import Wrench
import numpy as np

data_left = []
data_force = (0.0,0.0,0.0)

def left_finger_callback(msg):
	global data_left
	data_left = msg.data_array

def endpoint_callback(msg):
	global data_force
	data_force = (msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)

def handle_request(req):
	return (data_force, data_left)

def tactile_server():
	rospy.init_node('tactile_server')
	s = rospy.Service('tactile_service', tactile, handle_request)
	rospy.spin()

if __name__ == '__main__':
	rospy.Subscriber('/wsg_50_driver/left_finger_dsa_array', fingerDSAdata, left_finger_callback)
	rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState, endpoint_callback)
	tactile_server()