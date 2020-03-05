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
				 init_pos=True,
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
		self.force_upper = np.array([1, 2, 2])
		self.force_threshold = 15.0
		self.tactile_threshold = 40/3895

		self.start_pose = (np.array([-0.34070215,  0.07658594, -2.06358695,  0.62326658, -1.08047271,-1.6632061 , -3.86261129]), np.array([ 0.70886695, -0.30748498,  0.37799653]))
		self.goal_dir = fixed_goal

		self.z = 0.3774128
		self.previous_pose = self._get_endeffector_pose()[:3]
		self.ep_steps = 0
		self._max_episode_steps=max_episode_steps

		test = self.get_env_state()

		_,_,eepos = self.request_observation()
		print(eepos)
		print(test)
		if init_pos:
			self.init_pos()


	def _set_observation_space(self):
		# Tuple of (end eff force, tactile sensor readings array)
		# 14 x 6 tactile sensor reading (maybe need to double for both fingers??)
		# TODO: make the force reading bounds correct
		#self.observation_space = Tuple((Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64), 
		#								Box(low=0, high=3895, shape=(84,), dtype=np.int64)))
		self.observation_space = Box(low=0., high=1., shape=(87,), dtype=np.float32)

	def _set_action_space(self):
		# finger_pos
		self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float64)


	def _act(self, action):
		if self.action_mode == 'position':
			#angles,_,pose = self.request_observation()
			print("Action",action * self.gripper_pos_scale)
			self.set_gripper_pos(action * self.gripper_pos_scale)
			self._position_act(np.array([0,0.2,0])*self.position_action_scale)

	def _position_act(self, action):
		ee_pos = self._get_endeffector_pose()
		endeffector_pos = ee_pos[:3]
		self.previous_pose = endeffector_pos
		endeffector_pos[2] = self.z # fix z
		target_ee_pos = (endeffector_pos + action)
		target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
		orientation = (0.73276788, -0.68028504, 0.00153471, 0.01616033)
		target_ee_pos = np.concatenate((target_ee_pos, orientation))
		angles = self.request_ik_angles(target_ee_pos, self._get_joint_angles())
		self.send_angle_action(angles, target_ee_pos)


	def _get_obs(self):
		# get the reading from the end effector force and tactile sensor
		# TODO format the output into the Box 
		response = self.tactile()
		force, tactile = np.array(response.force), np.array(response.left_tactile)
		# scale the obs
		force = np.absolute(force / self.force_threshold)
		tactile = tactile / 3895
		#return np.array(response.force), np.array(response.left_tactile)
		return np.concatenate((force,tactile))

	def _get_info(self, action, observation, reward, done):
		info = {'gripper_action': action * self.gripper_pos_scale,
				'force_obs': observation[:3],
				'tactile_obs': observation[3:],
				'reward': reward,
				'done': done}

		return info

	def check_fail(self, obs, condition="both"):
		# check if force within the safe range

		#force_obs, tactile_obs = obs
		force_obs = obs[:3]
		tactile_obs = obs[3:]

		if condition == "force":
			# TODO: check the given force params
			force_obs = np.array(force_obs)
			result = (force_obs >= self.force_lower) * (force_obs <= self.force_upper)
			if result.prod() == 0:
				return True
		elif condition == "tactile":
			# this value can be set to something else if needed
			if tactile_obs.sum() <= 0:
				return True
		elif condition == "both":
			force_obs = np.array(force_obs)
			result = (force_obs > self.force_upper) 
			if result.sum():
				print("Force exceeded", force_obs)
				return True
			tactile_check = np.average(tactile_obs < self.tactile_threshold)
			max_val = np.max(tactile_obs)
			if tactile_check > 0.90:
				print("tactile below threshold", tactile_check)
				return True

		return False

	def step(self, action):
		self.ep_steps += 1
		self._act(action)
		observation = self._get_obs()
		# check if done
		if self.ep_steps >= self._max_episode_steps:
			done = True
		else:
			done = self.check_fail(observation)
		reward = self.compute_rewards(action, done)
		info = self._get_info(action, observation, reward, done)

		return observation, reward, done, info

	def reset(self):
		self._reset_robot()
		self.ep_steps = 0
		return self._get_obs()

	def get_diagnostics(self, paths, prefix=''):
		return OrderedDict()

	def compute_rewards(self, actions, done, binary=False):
		if binary:
			if done:
				return 0
			else:
				return 1
		else:
			if done:
				return 0
			else:
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
			self.set_gripper_pos(30)
			# reset
			#self.set_env_state((np.array([-0.48215625,  0.10807519, -2.44455576,  0.65573239, -0.70457029,-1.91778326, -3.90462208]), np.array([ 0.71286875, -0.32078603,  0.38858441])))
			self.set_env_state(self.start_pose)
			self.set_gripper_pos(0)

	def init_pos(self):
		self.set_gripper_pos(30)
		# reset sequence
		self._position_act([0.,0.,0.15])
		# neutral position
		self.set_env_state((np.array([ 1.90388665e-01, -1.24960935e+00,  4.02050791e-03,  2.18924403e+00,-2.76762199e+00, -7.03871071e-01, -4.71395683e+00]), np.array([0.40441269, 0.00732501, 0.21100701])))
		print("at neutral")
		# waypoints
		self.set_env_state((np.array([-0.07711425, -1.26722658, -0.61522853,  1.59733498, -2.93815041, -1.1669482 , -4.71084547]), np.array([ 0.32760957, -0.3180328 ,  0.48175633])))
		print("at pt1")
		self.set_env_state((np.array([-0.31150195, -0.17038867, -1.33267188,  0.34953126, -1.81241894,-1.62664163, -3.52767372]), np.array([ 0.71536082, -0.33902037,  0.43457955])))
		print("at pt2")
		for _ in range(5):
			self._position_act([ 0.7275, -0.3027451 ,  0.5] - self._get_endeffector_pose())
		self._position_act([0.,0.,-0.10])
		self.set_gripper_pos(0)                
	
	def set_gripper_pos(self, pos):
		self.gripper_pos(float(pos), 70.0)