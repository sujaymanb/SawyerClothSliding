from geometry_msgs.msg import Quaternion
import numpy as np
#JOINT_CONTROLLER_SETTINGS
JOINT_POSITION_SPEED = .1
JOINT_POSITION_TIMEOUT = .5

#JOINT INFO
JOINT_NAMES = ['right_j0',
               'right_j1',
               'right_j2',
               'right_j3',
               'right_j4',
               'right_j5',
               'right_j6'
               ]
LINK_NAMES = ['right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_l6', '_hand']
RESET_ANGLES = np.array(
    [0.74877149,  0.52912599, -1.42238283,  2.08878708, 0.4102627,-1.35198736,  0.34974024]   
    #[-0.26255566, -0.15996289, -1.43498826,  0.83371973,-1.77665627,-1.50944924, -4.1220293 ]
)

RESET_DICT = dict(zip(JOINT_NAMES, RESET_ANGLES))
POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=-0.7370773, y = 0.00671151, z=-0.67577171, w = 0.00215343
    #x=0.70925844, y=-0.7024014,   z=0.05931745, w=-0.00813613
)