from collections import OrderedDict
from gym.spaces import Box, Tuple
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from wsg_50_common.srv import Move
from std_srvs.srv import Empty
from sawyer_control.srv import tactile, angle_conv, endpoint_pose, tip_pose, angle_action
import numpy as np
import cv2
import rospy
from scipy.spatial.transform import Rotation as R

class SawyerSlideEnv(SawyerEnvBase):

	def __init__(self,
				 fix_goal=True,
				 fixed_goal=(0.53529716, 0.16153553, 0.37543553),
				 action_mode='position',
				 goal_low=None,
				 goal_high=None,
				 reset_free=False,
				 max_episode_steps=100,
				 position_action_scale=[0.1,0.1],
				 init_pos=True,
				 **kwargs
				 ):
		Serializable.quick_init(self, locals())
		
		SawyerEnvBase.__init__(self, action_mode=action_mode, 
							position_action_scale=position_action_scale,
							**kwargs)
		
		self.tactile = rospy.ServiceProxy('tactile_service', tactile, persistent=False)
		self.gripper_pos = rospy.ServiceProxy('/wsg_50_driver/move', Move, persistent=False)
		self.ack = rospy.ServiceProxy('/wsg_50_driver/ack', Empty, persistent=False)
		self.get_orient_action = rospy.ServiceProxy('angle_conv', angle_conv, persistent=False)
		self.endpoint_pose = rospy.ServiceProxy('endpoint_pose_server', endpoint_pose, persistent=False)
		self.tip_pose = rospy.ServiceProxy('tip_pose_server', tip_pose, persistent=False)
		
		self.gripper_scale = 3.5
		self.gripper_step = 0.1
		self.fixed_gripper = 0.5
		self.angle_scale = np.deg2rad(5)

		self.f_min = np.array([9.0, 9.0, 9.0])
		self.f_max = np.array([15.0, 10.0, 10.0])
		self.force_scale = 15.0 # default 15
		self.tactile_failure_threshold = 40

		#self.start_pose = (np.array([-0.39395702,  0.08154394, -1.79615045,  0.91780859, -1.33018267,-1.69333982, -4.21501446]), np.array([ 0.52982676, -0.39510188,  0.35705885]))
		# low
		#self.start_pose = (np.array([-0.13,  0.31 , -1.74 ,  1.23, -1.27,-1.21, -4.22]), np.array([ 0.50, -0.32,  0.15]))
		# horz
		self.start_pose = (np.array([-0.08822852,  0.17590527, -1.45440626,  1.75456834,  0.06563086,-1.84562397, -4.71270514]), np.array([ 0.52367145, -0.23880006,  0.22872107]))

		self.previous_pose = self.get_tip_pose()[:3]
		self.ep_steps = 0
		self._max_episode_steps=max_episode_steps

		# position at last reset
		self.rpos = None

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
		self.observation_space = Box(low=0., high=1., shape=(89,), dtype=np.float32)

	def _set_action_space(self):
		# x, angle
		self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float64)


	def _act(self, action):
		with np.printoptions(precision=3, suppress=True):
			print("Action",action)
		self._position_act(np.array([action[0]*self.position_action_scale[0], self.position_action_scale[1], 0., action[1]*self.angle_scale]))
		#self._position_act(np.array([action[0]*self.position_action_scale[0], 0., 0., action[1]*self.angle_scale]))

	def _position_act(self, action):
		ee_pos = self.get_tip_pose()
		endeffector_pos = ee_pos[:3]
		self.previous_pose = endeffector_pos
		target_ee_pos = (endeffector_pos + action[:3])
		target_ee_pos[2] = 0.2139
		target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
		_,_,pose = self.request_observation()
		current_o = pose[3:]
		orientation = self.orient_action(current_o,action[3],2)
		target_ee_pos = np.concatenate((target_ee_pos, orientation))

		# convert to hand pose if needed
		target_ee_pos = self.get_goal_pose(target_ee_pos.tolist())
		angles = self.request_ik_angles(target_ee_pos, self._get_joint_angles())
		self.send_angle_action_tip(angles, target_ee_pos)


	def get_goal_pose(self, pose):
		rospy.wait_for_service('endpoint_pose_server')
		try:
			resp = self.endpoint_pose(pose)
			return resp.ee_pose
		except rospy.ServiceException as e:
			print(e)


	def orient_action(self, current, angle_action, dim):
		if angle_action == 0.0:
			return current
		rospy.wait_for_service('angle_conv')
		try:
			resp = self.get_orient_action(current,angle_action,dim)
			return resp.orientation
		except rospy.ServiceException as e:
			print(e)


	def get_tip_pose(self):
		#rospy.wait_for_service('tip_pose_server')
		try:
			resp = self.tip_pose()
			return np.array(resp.tip_pose)
		except rospy.ServiceException as e:
			print(e)


	def _get_obs(self):
		# get the reading from the end effector force and tactile sensor
		response = self.tactile()
		force, tactile = np.array(response.force), np.array(response.left_tactile)
		
		# z pos and angle
		ee_pos = self.get_tip_pose()
		z = ee_pos[2]
		rot = R.from_quat(ee_pos[3:])
		r,p,y = rot.as_euler('xyz',degrees=True)

		# scale the obs
		force = np.absolute(force / self.force_scale)
		tactile = tactile / 3895
		angle = p / 30

		return np.concatenate((force,tactile,[z],[angle]))


	def _get_info(self,adj_label):
		info = {'gripper_width':self.fixed_gripper,
				'gripper_adj_label':adj_label}

		return info


	def check_fail(self, obs):
		# scale back obs
		force = obs[:3] * self.force_scale
		tactile = obs[3:87] * 3895
		label = None

		# < 90% cells less than fail_thresh
		if np.average(tactile < self.tactile_failure_threshold) < 0.90:
			if np.any(force > self.f_max):
				# increase gripper width
				self.fixed_gripper = min(self.gripper_scale, self.fixed_gripper + self.gripper_step*self.gripper_scale)
				print("Force exceeded, adjusting gripper %.3f mm"%self.fixed_gripper)
				label = 'increase'
			else:
				# if tactile is too low, decrease gripper width
				if (tactile > self.tactile_failure_threshold).shape[0] > 0:
					mean = np.mean(tactile[np.where(tactile > self.tactile_failure_threshold)])
				else:
					mean = 0
				coverage = np.sum(tactile > self.tactile_failure_threshold)

				#if coverage/84 < 0.30 and mean < 1250:
				if coverage/84 < 0.15 and mean < 1250:
					# decrease gripper width
					self.fixed_gripper = max(0.5, self.fixed_gripper - self.gripper_step*self.gripper_scale)
					print("grip low, adjusting gripper %.3f"%self.fixed_gripper)
					label = 'decrease'
				else:
					label = 'do_nothing'

		elif np.all(force < self.f_min):
				if self.fixed_gripper <= 0.5:
					label = 'do_nothing'
					return True,label
				else:
					# decrease gripper width
					self.fixed_gripper = max(0.5, self.fixed_gripper - self.gripper_step*self.gripper_scale)
					print("grip low, adjusting gripper %.3f mm"%self.fixed_gripper)
					label = 'decrease'
		else:
			label = 'do_nothing'

		self.set_gripper_pos(self.fixed_gripper)
		return False,label


	def step(self, action):
		self.ep_steps += 1
		self._act(action)
		observation = self._get_obs()
		fail,adj_label = self.check_fail(observation)
		#fail = False
		if self.ep_steps >= self._max_episode_steps:
			done = True
		else:
			done = fail
		reward = self.compute_rewards(action, fail, done, observation)
		info = self._get_info(adj_label)
		#info = None

		#print("\ncurrent z: %.3f"%observation[88]," z action: %.4f\n"%(action[1]*self.position_action_scale[1]))

		return observation, reward, done, info


	def reset(self):
		self._reset_robot()
		ret = None
		while ret is None:
			ret = self.calib_grasp_width()
			if ret is not None:
				self.fixed_gripper = ret
				break
			# manually reset cloth
			self.set_gripper_pos(60)
			input("Manually reset cloth...")			

		self.ep_steps = 0
		return self._get_obs()


	def get_diagnostics(self, paths, prefix=''):
		return OrderedDict()


	def compute_rewards(self, action, fail, done, obs, binary=False, use_action=False):
		#todo: change done to fail
		if binary:
			if fail:
				return 0
			else:
				return 1
		else:
			if fail:
				return 0
			else:
				# distance moved
				if use_action and not self.fix_hor:
					reward = (action[0]+1)*0.5*self.position_action_scale[0]
				elif use_action:
					reward = self.position_action_scale[0]
				else:
					ee_pos = self.get_tip_pose()[:3]
					reward = np.linalg.norm(ee_pos - self.previous_pose)

				return reward

	# not used
	def set_to_goal(self, goal):
		return


	# reset the environment
	def _reset_robot(self):
		if not self.reset_free:
			print("resetting")
			self.set_gripper_pos(60)
			self.set_gripper_pos(60)
			ee_pos = self.get_tip_pose()
			x,y,z = ee_pos[:3]
			self._position_act([x-0.1,0.,0.,0.])
			for _ in range(10):
				self.set_env_state((np.array([-0.27838477,  0.71916604, -1.83226275,  2.0155704 ,  0.4453418 ,-2.14861822, -4.71208477]), np.array([ 0.45134321, -0.26826239,  0.18772466])))
			for _ in range(10):
				self.set_env_state((np.array([-0.18840332,  0.70596194, -1.8959316 ,  1.85415041,  0.30685937,-1.88274312, -4.71208477]), np.array([ 0.51890278, -0.2535466 ,  0.16902347])))
			self.set_gripper_pos(4)
			for _ in range(10):
				self.set_env_state(self.start_pose)

			# save last reset pos
			_,_,eepos = self.request_observation()
			self.rpos = eepos[:3]


	# move arm back to height
	def reset_z(self, state):
		eepos = self.get_tip_pose()
		cpos = eepos[:3]
		cangle = np.deg2rad(state[88] * 30)

		#print("cangle rad",cangle)
		print("cpos z ",cpos[2])

		if cpos[2] < 0.025:
			print("reset z")
			self.set_gripper_pos(0.1)
			h = 0.075
			e = 999.0
			while e > 0.01:
				e = h - cpos[2]
				self._position_act([0,0,e,0]) 
				eepos = self.get_tip_pose()
				cpos = eepos[:3]
				
			#print(np.linalg.norm(e))

			self.set_gripper_pos(self.fixed_gripper)

			return self._get_obs()
		return None


	# find good grasp width for the cloth
	def calib_grasp_width(self, cov=0.2, exc=0.80, mmin=2300, mmax=3500):
		# adjust the grasp smaller and smaller until threshold
		scale = self.gripper_scale
		width = 2*scale/2+0.5

		fail = 0
		fail_max = 3
		manual = False

		while True:
			self.set_gripper_pos(width)
			obs = self._get_obs()
			tactile = obs[3:87] * 3895

			mean = np.mean(tactile[np.where(tactile > 100)])
			coverage = np.sum(tactile > 100)
			excess = np.sum(tactile > mmax)

			with np.printoptions(precision=3, suppress=True):
				print("width:",round(width,2),"mean:",round(mean,2),"coverage %:",round(coverage/84,2),"excess %:",round(excess/coverage,2))
			# defaults: 3000 > mean > 1250, coverage %: 30 %, excess %: 40
			if coverage/84 > cov and excess/coverage < exc and mean < mmax and mean > mmin:
				break
			if width > 0.5:
				width -= 0.1
			else:
				fail += 1

				if coverage/84 < cov and mean > mmin:
					width = 2*scale/2+0.5
					self.set_gripper_pos(10)
					self._position_act([0.,0.,-0.01,0.])
				elif mean < mmin:
					width = 2*scale/2+0.5
					self.set_gripper_pos(10)
					self._position_act([0.,0.,0.01,0.])
				else:
					fail = 3

			if fail >= 3:
				manual = True
				break
				

		if manual:
			return None
		return width



	# go to neutral then to init pos above cloth
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
		        
	
	def set_gripper_pos(self, pos):
		rospy.wait_for_service('/wsg_50_driver/move')
		try:
			resp = self.gripper_pos(float(pos), 70.0)
			#print(resp)
			if resp.error > 0:
				rospy.wait_for_service('/wsg_50_driver/ack')
				ackresp = self.ack()
				#print('acked')
				rospy.wait_for_service('/wsg_50_driver/move')
				resp = self.gripper_pos(float(pos), 70.0)
				#print('reattempt')
				rospy.wait_for_service('/wsg_50_driver/ack')
				ackresp = self.ack()

		except rospy.ServiceException as e:
			print(e)
		


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