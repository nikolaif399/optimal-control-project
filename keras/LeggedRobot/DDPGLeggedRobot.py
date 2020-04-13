import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

from rl.agents.ddpg import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory

from envs.spirit_env import SpiritEnv

def main():

	np.random.seed(123)

	# Get the environment and extract the number of actions.
	train_env = SpiritEnv(False)
	train_env.seed(123)

	n_act = train_env.action_space.shape[0]
	obs_shape = train_env.observation_space.shape # 36 x 1

	# Build Policy Model (Actor)
	actor = Sequential()
	actor.add(Flatten(input_shape=(1,) + obs_shape))
	actor.add(Dense(100))
	actor.add(Activation('relu'))
	actor.add(Dense(100))
	actor.add(Activation('relu'))
	actor.add(Dense(100))
	actor.add(Activation('relu'))
	actor.add(Dense(n_act))
	actor.add(Activation('tanh'))
	plot_model(actor, to_file='plots/actor.png', show_shapes=True)
	print(actor.summary())

	# Build Q-Function Model (Critic)
	action_input = Input(shape=(n_act,), name='action_input')
	observation_input = Input(shape=(1,) + train_env.observation_space.shape, name='observation_input')
	flattened_observation = Flatten()(observation_input)
	x = Dense(200)(flattened_observation)
	x = Activation('relu')(x)
	x = Concatenate()([x, action_input])
	x = Dense(200)(x)
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
	actor.save('models/legged_robot_ddpg_weights.h5', overwrite=True)


	#test_env = SpiritEnv(True)
	#test_env.seed(123)
	#agent.test(test_env, nb_episodes=5, visualize=False)

if __name__ == "__main__":
	main()