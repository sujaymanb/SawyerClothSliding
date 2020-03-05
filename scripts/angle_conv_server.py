#!/usr/bin/env python
import rospy
import tf
from sawyer_control.srv import angle_conv
import numpy as np

def get_orient(msg):
    print(msg)
    current = msg.current_orientation
    action = msg.action

    r,p,y = tf.transformations.euler_from_quaternion(current)
    print('rpy',r,p,y)
    p += action
    # limit within fixed range (30 deg +/- of vertically down)
    p = np.clip(p,-0.54783780694,0.49936019306)

    orientation = tf.transformations.quaternion_from_euler(r,p,y)
    print('ori',orientation)
    return {'orientation':orientation}
    
def angle_conv_server():
    rospy.init_node('angle_conv', anonymous=True)
    s = rospy.Service('angle_conv', angle_conv, get_orient)
    rospy.spin()

if __name__ == '__main__':
    angle_conv_server()