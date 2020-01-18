import rospy
from wsg_50_common.srv import Move
import wsg_50_common.srv
from sawyer_control.srv import tactile

rospy.wait_for_service('/wsg_50_driver/move')
gripper_pos = rospy.ServiceProxy('/wsg_50_driver/move', Move)
#req = wsg_50_common.srv.Move(30.0, 10.0)
#resp = gripper_pos(req)
resp = gripper_pos(40.0, 20.0)

rospy.wait_for_service('tactile_service')
obs = rospy.ServiceProxy('tactile_service', tactile, persistent=True)
resp = obs()
print(resp)