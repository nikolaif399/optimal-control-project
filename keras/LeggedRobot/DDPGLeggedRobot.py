import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

from rl.agents.ddpg import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory

from envs.legged_robot_env import LeggedRobotEnv

test_actor = True

def main():

	np.random.seed(123)

	# Get the environment and extract the number of actions.
	train_env = LeggedRobotEnv(False)
	train_env.seed(123)

	n_act = train_env.action_space.shape[0]
	obs_shape = train_env.observation_space.shape # 36 x 1

	# Build Policy Model (Actor)
	actor = Sequential()
	actor.add(Flatten(input_shape=(1,) + obs_shape))
	actor.add(Dense(400))
	actor.add(Activation('relu'))
	actor.add(Dense(300))
	actor.add(Activation('relu'))
	actor.add(Dense(n_act))
	actor.add(Activation('sigmoid'))
	plot_model(actor, to_file='plots/actor.png', show_shapes=True)
	print(actor.summary())

	if (test_actor):
		actor.load_weights("weights/legged_robot_ddpg_weights_actor.h5f")
		x = np.array([[-6.23736334e-08],
					 [-2.03309437e-07],
					 [ 1.97399179e-01],
					 [-6.23736334e-06],
					 [-2.03309437e-05],
					 [-1.73903108e-01],
					 [ 5.55218035e-03],
					 [-1.73856866e-03],
					 [-4.28090865e-06],
					 [ 3.61358288e-01],
					 [-1.15324906e-01],
					 [ 4.85251444e-05],
					 [-1.62067152e-07],
					 [ 0.00000000e+00],
					 [-7.86601299e-08],
					 [ 0.00000000e+00],
					 [ 4.16077794e-07],
					 [ 2.90905541e-07],
					 [ 5.48785523e-07],
					 [ 0.00000000e+00],
					 [ 1.84871025e-06],
					 [-6.88562149e-08],
					 [-4.92839454e-10],
					 [ 0.00000000e+00],
					 [-7.65393652e-06],
					 [ 0.00000000e+00],
					 [-3.43654177e-06],
					 [ 0.00000000e+00],
					 [ 1.98390255e-05],
					 [ 1.37557055e-05],
					 [ 2.58356920e-05],
					 [ 0.00000000e+00],
					 [ 8.71664840e-05],
					 [-3.18024553e-06],
					 [-2.19911729e-08],
					 [ 0.00000000e+00],
					 [ 5.52796125e-01],
					 [ 9.95747209e+00],
					 [ 1.06311560e+01],
					 [ 9.13203239e+00],
					 [-1.19774938e-01],
					 [ 9.58176970e+00],
					 [ 9.57453370e+00],
					 [ 4.69361544e-01],
					 [-2.52103209e-01],
					 [ 3.65097523e-01],
					 [ 2.15576887e-01],
					 [-1.06084347e-01]])
		#scale = 1000000000000000000000
		#x = scale*np.random.randn(1,1,48,1)
		x = x.reshape((1,1,48,1))
		action = actor.predict(x)
		print(action)
		return

	# Build Q-Function Model (Critic)
	action_input = Input(shape=(n_act,), name='action_input')
	observation_input = Input(shape=(1,) + train_env.observation_space.shape, name='observation_input')
	flattened_observation = Flatten()(observation_input)
	x = Dense(400)(flattened_observation)
	x = Activation('relu')(x)
	x = Concatenate()([x, action_input])
	x = Dense(300)(x)
	x = Activation('relu')(x)
	x = Dense(1)(x)
	x = Activation('linear')(x)
	critic = Model(inputs=[action_input, observation_input], outputs=x)
	plot_model(critic, to_file='plots/critic.png', show_shapes=True)
	print(critic.summary())

	# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
	# even the metrics!
	memory = SequentialMemory(limit=100000, window_length=1)
	random_process = OrnsteinUhlenbeckProcess(size=n_act, theta=.15, mu=0., sigma=.3)
	agent = DDPGAgent(nb_actions=n_act, actor=actor, critic=critic, critic_action_input=action_input,
	                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
	                  random_process=random_process, gamma=.99, target_model_update=1e-3)
	agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


	agent.fit(train_env, nb_steps=500, visualize=False, verbose=2)
	agent.save_weights('weights/legged_robot_ddpg_weights.h5f', overwrite=True)


	#test_env = LeggedRobotEnv(True)
	#test_env.seed(123)
	#agent.test(test_env, nb_episodes=5, visualize=False)

if __name__ == "__main__":
	main()