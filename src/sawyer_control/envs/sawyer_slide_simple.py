from collections import OrderedDict
from gym.spaces import Box, Tuple
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from wsg_50_common.srv import Move
from sawyer_control.srv import tactile
import numpy as np
import cv2
import rospy

class SawyerSlideEnv(SawyerEnvBase):

	def __init__(self,
				 fix_goal=True,
				 fixed_goal=(1,0,0),
				 action_mode='position',
				 goal_low=None,
				 goal_high=None,
				 reset_free=False,
				 **kwargs
				 ):
		Serializable.quick_init(self, locals())
		SawyerEnvBase.__init__(self, action_mode=action_mode, **kwargs)
		self.tactile = rospy.ServiceProxy('tactile_service', tactile, persistent=False)
		self.gripper_pos = rospy.ServiceProxy('/wsg_50_driver/move', Move, persistent=False)
		self.gripper_pos_scale = 50

		# TODO: tune these upper and lower thresholds
		self.force_lower = np.array([0, 0, 0])
		self.force_upper = np.array([100, 100, 100])

		self.goal_dir = fixed_goal


	def _set_observation_space(self):
		# Tuple of (end eff force, tactile sensor readings array)
		# 14 x 6 tactile sensor reading (maybe need to double for both fingers??)
		# TODO: make the force reading bounds correct
		self.observation_space = Tuple((Box(low=0, high=np.inf, shape=(3,), dtype=np.float64), 
										Box(low=0, high=3895, shape=(84,), dtype=np.int64)))


	def _set_action_space(self):
		# finger_pos
		self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float64)


	def _act(self, action):
		if self.action_mode == 'position':
			self.set_gripper_pos(action * self.gripper_pos_scale)
			#self._position_act(self.goal_dir - self._get_endeffector_pose())


	def _get_obs(self):
		# get the reading from the end effector force and tactile sensor
		# TODO format the output into the Box 
		response = self.tactile()
		return np.array(response.force), np.array(response.left_tactile)

	def check_fail(self, obs, condition="force"):
		# check if force within the safe range

		force_obs, tactile_obs = obs

		if condition == "force":
			# TODO: check the given force params
			force_obs = np.array(force_obs)
			result = (force_obs >= self.force_lower) * (force_obs <= self.force_upper)
			if result.prod() == 1:
				return True
		elif condition == "tactile":
			# this value can be set to something else if needed
			if tactile_obs.sum <= 0:
				return True

		return False

	def step(self, action):
		self._act(action)
		observation = self._get_obs()
		reward = self.compute_rewards(action, observation, self._state_goal)
		info = self._get_info()
		# check if done
		done = self.check_fail(observation)

		return observation, reward, done, info

	def reset(self):
		self._reset_robot()
		return self._get_obs()

	def get_diagnostics(self, paths, prefix=''):
		return OrderedDict()

	def compute_rewards(self, actions, obs, goals):
		# TODO: compute rewards
		return 5

	# not sure why this is needed
	def set_to_goal(self, goal):
		return


	# TODO: test custom ROS functions
	def _reset_robot(self):
		if not self.reset_free:
			#self.open_gripper()
			#for _ in range(5):
				#self._position_act(self.pos_control_reset_position - self._get_endeffector_pose())
			self.set_gripper_pos(0)                

	def open_gripper(self):
		self.set_gripper_pos(50.0)
	
	def set_gripper_pos(self, pos):
		self.gripper_pos(float(pos), 70.0)