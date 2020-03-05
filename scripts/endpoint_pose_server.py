#!/usr/bin/env python
from sawyer_control.srv import *
import rospy
from geometry_msgs.msg import PoseStamped 
from tf.transformations import *
import tf

def transform(msg):
    x,y,z,qx,qy,qz,qw = msg.input_pose
    endpoint_pose = [x,y,z,qx,qy,qz,qw]
    
    endpoint_pose = []

    if tf_lis.frameExists("base") and tf_lis.frameExists("gripper_tip"):
        print("converting from gripper_tip to right_hand")
        t = tf_lis.getLatestCommonTime("base", "gripper_tip")
        
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'base'
        base_pose.header.stamp = t
        base_pose.pose.position.x = x
        base_pose.pose.position.y = y
        base_pose.pose.position.z = z
        base_pose.pose.orientation.x = qx
        base_pose.pose.orientation.y = qy
        base_pose.pose.orientation.z = qz
        base_pose.pose.orientation.w = qw

        # convert to tip frame
        tip_frame_pose = tf_lis.transformPose('gripper_tip',base_pose)

        # offset to convert to right_hand frame
        tip_frame_pose.pose.position.z -= 0.13

        # convert back to base frame
        base_pose = tf_lis.transformPose('base',tip_frame_pose)
        endpoint_pose = [base_pose.pose.position.x,
                         base_pose.pose.position.y,
                         base_pose.pose.position.z,
                         base_pose.pose.orientation.x,
                         base_pose.pose.orientation.y,
                         base_pose.pose.orientation.z,
                         base_pose.pose.orientation.w]
    else:
        endpoint_pose = [x,y,z,qx,qy,qz,qw]
    
    return {'ee_pose':endpoint_pose}

def endpoint_pose_server():
    rospy.init_node('endpoint_pose_server', anonymous=True)
    global tf_lis
    tf_lis = tf.TransformListener()
    s = rospy.Service('endpoint_pose_server', endpoint_pose, transform)
    rospy.spin()

if __name__ == '__main__':
    endpoint_pose_server()