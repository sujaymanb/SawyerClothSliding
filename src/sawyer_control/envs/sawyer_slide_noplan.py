from collections import OrderedDict
from gym.spaces import Box, Tuple
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from wsg_50_common.srv import Move
from sawyer_control.srv import tactile, angle_conv, endpoint_pose, tip_pose, angle_action
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
		self.get_orient_action = rospy.ServiceProxy('angle_conv', angle_conv, persistent=False)
		self.endpoint_pose = rospy.ServiceProxy('endpoint_pose_server', endpoint_pose, persistent=False)
		self.tip_pose = rospy.ServiceProxy('tip_pose_server', tip_pose, persistent=False)

		self.gripper_pos_scale = 10
		self.angle_scale = np.deg2rad(5) # degrees

		self.force_lower = np.array([0, 0, 0])
		self.force_upper = np.array([1, 2, 2])
		self.force_threshold = 15.0
		self.tactile_threshold = 40/3895

		#self.start_pose = (np.array([-0.40968263,  0.06017188, -1.89665329,  0.72006935, -1.22444236,-1.69499218, -4.01992989]), np.array([ 0.61895293, -0.38413632,  0.37347302]))
		self.start_pose = (np.array([-0.39395702,  0.08154394, -1.79615045,  0.91780859, -1.33018267,-1.69333982, -4.21501446]), np.array([ 0.52982676, -0.39510188,  0.35705885]))

		self.goal_dir = fixed_goal

		#self.z = 0.3774128
		#self.y = 0.07658594
		#TODO: fix X
		self.previous_pose = self.get_tip_pose()[:3]
		self.ep_steps = 0
		self._max_episode_steps=max_episode_steps

		# debug print
		#prev_t = rospy.Time.now()
		#while True:
		#	_,_,eepos = self.request_observation()
		#	eepos = self.get_tip_pose()
		#	eepos = self.get_env_state()
		#	print(eepos)
		#	curr_t = rospy.Time.now()
		#	print(curr_t-prev_t)
		#	prev_t = curr_t

		if init_pos:
			self.init_pos()


	def _set_observation_space(self):
		# (x,y,z) force readings
		# 14 x 6 tactile sensor reading (maybe need to double for both fingers??)
		self.observation_space = Box(low=0., high=1., shape=(87,), dtype=np.float32)

	def _set_action_space(self):
		# gripper_pos, (y,z), ee_angle
		self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float64)


	def _act(self, action):
		if self.action_mode == 'position':
			print("Action",action)
			#self.set_gripper_pos((action[0] + 1) * self.gripper_pos_scale * 0.5)
			#fix gripper
			self.set_gripper_pos(2.4)
			#self._position_act(np.array([0., np.absolute(action[1]*self.position_action_scale), action[2]*self.position_action_scale, action[3]*self.angle_scale]))
			self._position_act(np.array([0., (action[1]+1)*0.5*self.position_action_scale, action[2]*self.position_action_scale, action[3]*self.angle_scale]))

	def _position_act(self, action):
		ee_pos = self.get_tip_pose()
		endeffector_pos = ee_pos[:3]
		self.previous_pose = endeffector_pos
		target_ee_pos = (endeffector_pos + action[:3])
		target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
		_,_,pose = self.request_observation()
		current_o = pose[3:]
		#current_o = (0.73276788, -0.68028504, 0.00153471, 0.01616033)
		orientation = self.orient_action(current_o,action[3])
		target_ee_pos = np.concatenate((target_ee_pos, orientation))

		# convert to hand pose if needed
		target_ee_pos = self.get_goal_pose(target_ee_pos.tolist())
		#print(target_ee_pos)

		angles = self.request_ik_angles(target_ee_pos, self._get_joint_angles())
		self.send_angle_action_tip(angles, target_ee_pos)


	def get_goal_pose(self, pose):
		#print("waiting for goal pose...")
		rospy.wait_for_service('endpoint_pose_server')
		try:
			resp = self.endpoint_pose(pose)
			return resp.ee_pose
		except rospy.ServiceException as e:
			print(e)
		#print("got goal pose")


	def orient_action(self, current, angle_action):
		if angle_action == 0.0:
			return current
		rospy.wait_for_service('angle_conv')
		try:
			resp = self.get_orient_action(current,angle_action)
			return resp.orientation
		except rospy.ServiceException as e:
			print(e)


	def get_tip_pose(self):
		#print("Waiting for tip pose")
		rospy.wait_for_service('tip_pose_server')
		try:
			resp = self.tip_pose()
			return np.array(resp.tip_pose)
		except rospy.ServiceException as e:
			print(e)
		#print("Got tip pose")



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
				current_pose = self.get_tip_pose()[:3]
				#d = np.linalg.norm(current_pose - self.previous_pose)
				d = (current_pose[1] - self.previous_pose[1])
				return d

	# not used
	def set_to_goal(self, goal):
		return


	def _reset_robot(self):
		if not self.reset_free:
			print("resetting")
			self.set_gripper_pos(60)
			ee_pos = self.get_tip_pose()
			x,y,z = ee_pos[:3]
			self._position_act([0.,0.,0.40-z,0.])
			for _ in range(2):
				self.set_env_state((np.array([-0.40176561, -0.0182959 , -2.02688861,  0.76752341, -1.21615326,-1.84398437, -4.11705875]), np.array([ 0.59614933, -0.34725615,  0.46249017])))
			for _ in range(2):
				self.set_env_state(self.start_pose)
			#self._position_act([0.,0.,-0.02,0.])
			self.set_gripper_pos(0)

	def init_pos(self):
		print("moving to Init pose")
		self.set_gripper_pos(60)
		# reset sequence
		self._position_act([0.,0.,0.15,0.])
		# neutral position
		self.set_env_state((np.array([ 1.90388665e-01, -1.24960935e+00,  4.02050791e-03,  2.18924403e+00,-2.76762199e+00, -7.03871071e-01, -4.71395683e+00]), np.array([0.40441269, 0.00732501, 0.21100701])))
		print("at neutral")
		# waypoints
		self.set_env_state((np.array([-0.07711425, -1.26722658, -0.61522853,  1.59733498, -2.93815041, -1.1669482 , -4.71084547]), np.array([ 0.32760957, -0.3180328 ,  0.48175633])))
		print("at pt1")
		for _ in range(2):
			self.set_env_state((np.array([-0.42582616, -0.20270801, -1.84254885,  0.74962986, -1.43295407,-1.86295116, -4.19389248]), np.array([ 0.5272882 , -0.37081248,  0.5404008 ])))
		print("at pt2")
		#self._position_act([0.,0.,-0.10,0.])
		self.set_gripper_pos(0)                
	
	def set_gripper_pos(self, pos):
		self.gripper_pos(float(pos), 70.0)


	def send_angle_action_tip(self, action, target):
		self.request_angle_action_tip(action, target)


	def request_angle_action_tip(self, angles, pos):
		dist = np.linalg.norm(self.get_tip_pose()[:3] - pos[:3])
		duration = dist/self.max_speed
		rospy.wait_for_service('angle_action')
		try:
			execute_action = rospy.ServiceProxy('angle_action', angle_action, persistent=True)
			execute_action(angles, duration)
			return None
		except rospy.ServiceException as e:
			print(e)