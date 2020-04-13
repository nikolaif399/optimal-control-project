#from keras.models import load_model
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pybullet_data
import time

from envs.spirit_env import SpiritEnv

# Load trained policy
#policy = load_model('models/legged_robot_ddpg_weights.h5')

# Setup PyBullet Simulation

env = SpiritEnv(True,False)
env.seed(123)
env.reset()

hip_kp = 0.5
hip_kd = 0.05
knee_kp = 0.5 # Action signal per error radian
knee_kd = 0.05

while(env.stepCount < 5000):
	start = time.time()

	#hip_action = env.hip_nom_pos * np.ones((4,1))
	#knee_action = env.knee_nom_pos * np.ones((4,1))

	joint_pos = env.getJointPositions()
	abd_pos = joint_pos[0:4,:]
	hip_pos = joint_pos[4:8,:]
	knee_pos = joint_pos[8:12,:]

	knee_error = env.knee_nom_pos - knee_pos
	hip_error = env.hip_nom_pos - hip_pos

	joint_vel = env.getJointVelocities()
	abd_vel = joint_vel[0:4,:]
	hip_vel = joint_vel[4:8,:]
	knee_vel = joint_vel[8:12,:]

	hip_derror = -hip_vel
	knee_derror = -knee_vel

	hip_action = hip_kp * hip_error + hip_kd * hip_derror
	knee_action = knee_kp * knee_error + knee_kd * knee_derror

	action = np.vstack((hip_action, knee_action))
	env.step(action)

	appliedTorques = env.getJointTorques()
	#print("Hip Torques: ")
	#print(appliedTorques[4:8,:].T)

	#print("Knee Torques: ")
	#print(appliedTorques[8:12,:].T)

	left = env.tstep - (time.time() - start)
	if left > 0:
		time.sleep(left)