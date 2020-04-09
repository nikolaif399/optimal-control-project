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

np.random.seed(123)

# Get the environment and extract the number of actions.
env = LeggedRobotEnv(True)
env.seed(123)
env.reset()

n_act = env.action_space.shape[0]
obs_shape = env.observation_space.shape # 36 x 1

# Build Policy Model (Actor)
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + obs_shape))
actor.add(Dense(400, activation='relu'))
actor.add(Dense(300, activation='relu'))
actor.add(Dense(n_act, activation='linear'))
plot_model(actor, to_file='plots/actor.png', show_shapes=True)
print(actor.summary())

# Build Q-Function Model (Critic)
action_input = Input(shape=(n_act,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400, activation='relu')(flattened_observation)
x = Concatenate()([x, action_input])
x = Dense(300, activation='relu')(x)
x = Dense(1, activation='linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
plot_model(critic, to_file='plots/critic.png', show_shapes=True)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
print(n_act)
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=n_act, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=n_act, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


agent.fit(env, nb_steps=100, visualize=False, verbose=2)
agent.save_weights('weights/legged_robot_ddpg_weights.h5f', overwrite=True)
#agent.test(env, nb_episodes=5, visualize=True)
