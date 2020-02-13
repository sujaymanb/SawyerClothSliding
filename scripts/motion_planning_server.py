#!/usr/bin/env python
import rospy
import tf
from sawyer_control.srv import move_to_pose, move_to_angle
import numpy as np
from sawyer_control.pd_controllers.motion_planning import MotionPlanner

mp = MotionPlanner()

def handle_pose(msg):
    print(list(msg.pose))
    mp.move_to_pose(list(msg.pose))
    return True

def handle_angle(msg):
    mp.move_to_joint(list(msg.angles))
    return True
    
def motion_planning_server():
    rospy.init_node('mp_server', anonymous=True)
    s1 = rospy.Service('move_to_pose', move_to_pose, handle_pose)
    s2 = rospy.Service('move_to_angle', move_to_angle, handle_angle)
    rospy.spin()

if __name__ == '__main__':
    motion_planning_server()