0#from keras.models import load_model
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pybullet_data
import time

from envs.spirit_env import SpiritEnv

# Load trained policy
#policy = load_model('models/legged_robot_ddpg_weights.h5')

# Setup PyBullet Simulation

env = SpiritEnv(True,True)
env.seed(123)
env.reset()

while(env.stepCount < 5000):
	start = time.time()

	hip_action = env.hip_nom_pos * np.ones((4,1))
	knee_action = env.knee_nom_pos * np.ones((4,1))
	
	action = np.vstack((hip_action, knee_action))
	env.step(action)

	appliedTorques = env.getJointTorques()
	print("Hip Torques: ")
	print(appliedTorques[4:8,:].T)

	print("Knee Torques: ")
	print(appliedTorques[8:12,:].T)

	left = env.tstep - (time.time() - start)
	if left > 0:
		time.sleep(left)