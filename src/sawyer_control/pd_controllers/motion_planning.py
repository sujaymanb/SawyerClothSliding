import sys
import rospy
import moveit_commander
#from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

class MotionPlanner():
	def __init__(self):
		moveit_commander.roscpp_initialize(sys.argv)
		self.scene = moveit_commander.PlanningSceneInterface()
		self.group = moveit_commander.MoveGroupCommander("right_arm")

	def move_to_pose(self, target_pose):
		self.group.clear_pose_targets()
		self.group.set_pose_target(target_pose)
		self.group.set_planning_time(10)
		#plan = self.group.plan()
		#self.group.execute(plan, wait=True)
		plan = self.group.go(wait=True)
		self.group.stop()
		self.group.clear_pose_targets()

	def move_to_joint(self, target_joints):
		plan = self.group.go(target_joints, wait=True)
		self.group.stop()
