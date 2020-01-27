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
				 fixed_goal=(0.53529716, 0.16153553, 0.37543553),
				 action_mode='position',
				 goal_low=None,
				 goal_high=None,
				 reset_free=False,
				 max_episode_steps=10,
				 position_action_scale=0.1,
				 **kwargs
				 ):
		Serializable.quick_init(self, locals())
		SawyerEnvBase.__init__(self, action_mode=action_mode, 
							position_action_scale=position_action_scale,
			 				**kwargs)
		self.tactile = rospy.ServiceProxy('tactile_service', tactile, persistent=False)
		self.gripper_pos = rospy.ServiceProxy('/wsg_50_driver/move', Move, persistent=False)
		self.gripper_pos_scale = 10

		# TODO: tune these upper and lower thresholds
		self.force_lower = np.array([0, 0, 0])
		self.force_upper = np.array([100, 100, 100])

		#self.start_pose = (np.array([ 0.35342872, -0.65991211, -0.82064646,  1.42971587, -2.43175673,-1.10287106, -3.65176177]), np.array([ 0.49755925, -0.12712517,  0.34312445]))
		self.start_pose = (np.array([-0.37102735, -0.14880371, -1.11035252,  0.88896096, -1.90269244, -1.2040664 , -4.24795008]), np.array([ 0.50736415, -0.47188753,  0.27946076]))
		self.goal_dir = fixed_goal
		#_,_,eepos = self.request_observation()
		#print(eepos)

		self.z = 0.27946076
		self.previous_pose = self._get_endeffector_pose()[:3]
		self.ep_steps = 0
		self._max_episode_steps=max_episode_steps

		#test = self.get_env_state()
		#print(test)


	def _set_observation_space(self):
		# Tuple of (end eff force, tactile sensor readings array)
		# 14 x 6 tactile sensor reading (maybe need to double for both fingers??)
		# TODO: make the force reading bounds correct
		#self.observation_space = Tuple((Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64), 
		#								Box(low=0, high=3895, shape=(84,), dtype=np.int64)))
		self.observation_space = Box(low=-np.inf, high=np.inf, shape=(87,), dtype=np.float32)

	def _set_action_space(self):
		# finger_pos
		self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float64)


	def _act(self, action):
		if self.action_mode == 'position':
			self.set_gripper_pos(action * self.gripper_pos_scale)
			self._position_act(np.array([0,0.2,0])*self.position_action_scale)

	def _position_act(self, action):
		ee_pos = self._get_endeffector_pose()
		endeffector_pos = ee_pos[:3]
		self.previous_pose = endeffector_pos
		endeffector_pos[2] = self.z # fix z
		target_ee_pos = (endeffector_pos + action)
		target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
		orientation = ee_pos[3:]
		target_ee_pos = np.concatenate((target_ee_pos, orientation))
		angles = self.request_ik_angles(target_ee_pos, self._get_joint_angles())
		self.send_angle_action(angles, target_ee_pos)


	def _get_obs(self):
		# get the reading from the end effector force and tactile sensor
		# TODO format the output into the Box 
		response = self.tactile()
		force, tactile = np.array(response.force), np.array(response.left_tactile)
		# scale the obs
		force = force / 20
		tactile = tactile / 3895
		#return np.array(response.force), np.array(response.left_tactile)
		return np.concatenate((force,tactile))

	def check_fail(self, obs, condition="none"):
		# check if force within the safe range

		#force_obs, tactile_obs = obs
		force_obs = obs[:3]
		tactile_obs = obs[3:]

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
		self.ep_steps += 1
		self._act(action)
		observation = self._get_obs()
		reward = self.compute_rewards(action, observation, self._state_goal)
		info = self._get_info()
		# check if done
		if self.ep_steps >= self._max_episode_steps:
			done = True
		else:
			done = self.check_fail(observation)

		return observation, reward, done, info

	def reset(self):
		self._reset_robot()
		self.ep_steps = 0
		return self._get_obs()

	def get_diagnostics(self, paths, prefix=''):
		return OrderedDict()

	def compute_rewards(self, actions, obs, goals):
		# distance moved
		current_pose = self._get_endeffector_pose()[:3]
		d = np.linalg.norm(current_pose - self.previous_pose)
		return d

	# not sure why this is needed
	def set_to_goal(self, goal):
		return


	# TODO: test custom ROS functions
	def _reset_robot(self):
		if not self.reset_free:
			print("resetting")
			self.set_gripper_pos(10)
			#for _ in range(5):
			#	self._position_act(self.start_pose - self._get_endeffector_pose())
			for _ in range(5):
				self.set_env_state(self.start_pose)
			self.set_gripper_pos(0)                
	
	def set_gripper_pos(self, pos):
		self.gripper_pos(float(pos), 70.0)