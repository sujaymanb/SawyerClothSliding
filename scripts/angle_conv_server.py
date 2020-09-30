#!/usr/bin/env python
import rospy
import tf
from sawyer_control.srv import angle_conv
import numpy as np

rpy_min = np.array([-999.,-0.785398,-2.35619])
rpy_max = np.array([999., 0.785398, -0.785398])

def get_orient(msg):
    print(msg)
    current = msg.current_orientation
    action = msg.action
    dim = msg.dim

    r,p,y = tf.transformations.euler_from_quaternion(current)
    rpy = [r,p,y]

    print(r,p,y)
    # limit to +/- 45 deg
    print(rpy[dim]+action)
    #a = np.clip(rpy[dim]+action,-0.785398,0.785398)
    rpy[dim] += action
    #rpy = np.clip(rpy,rpy_min,rpy_max)
    r,p,y = rpy
    orientation = tf.transformations.quaternion_from_euler(r,p,y)
    #print('ori',orientation)
    return {'orientation':orientation}
    
def angle_conv_server():
    rospy.init_node('angle_conv', anonymous=True)
    s = rospy.Service('angle_conv', angle_conv, get_orient)
    rospy.spin()

if __name__ == '__main__':
    angle_conv_server()