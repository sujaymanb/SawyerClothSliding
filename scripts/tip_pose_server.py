#!/usr/bin/env python
from sawyer_control.srv import *
import rospy
from tf import TransformListener
from geometry_msgs.msg import PoseStamped 
from tf.transformations import *

def get_tip_pose(unused):
	try:
		t = tf_lis.getLatestCommonTime("base", "gripper_tip")
		pos, orientation = tf_lis.lookupTransform("base", "gripper_tip", t)
		tip_pose = pos + orientation
		return {'tip_pose':tip_pose}
	except Exception as e:
		print("could not get tip pose")
		pass
	

def tip_pose_server():
	rospy.init_node('tip_pose_server', anonymous=True)

	global tf_lis
	tf_lis = TransformListener()

	s = rospy.Service('tip_pose_server', tip_pose, get_tip_pose)
	rospy.spin()

if __name__ == "__main__":
	tip_pose_server()