import gym
from gym import spaces
import numpy as np

class AirHockeyEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(AirHockeyEnv, self).__init__()
    
    # Define Air Hockey Physical Environment
    massStriker = 1.0
    massPuck = 0.5
    radiusStriker = 0.08
    radiusPuck = 0.06
    maxForce = 30 # Maximum force applied to striker

    tableWidth = 2 # X Dimension
    tableHeight = 1 # Y Dimension
    #frictionCoeff = 0.0 # Friction between puck and surface

    # Define action and observation space
    # They must be gym.spaces objects
    self.action_space = spaces.Box(low=-maxForce, high=maxForce, shape = (2,1), dtype=np.float32)

    self.min_pos = np.array([0,0]).reshape((2,1))
    self.max_pos = np.array([tableWidth, tableHeight]).reshape((2,1))
    self.max_striker_vel = np.array([3,3]).reshape((2,1))
    self.max_puck_vel = np.array([6,6]).reshape((2,1))
    self.min_obs = np.vstack((self.min_pos, -self.max_striker_vel, self.min_pos, -self.max_puck_vel))
    self.max_obs = np.vstack((self.max_pos, self.max_striker_vel, self.max_pos, self.max_puck_vel))

    self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, dtype=np.float32)

  def step(self, action):
    # Execute one time step within the environment
    # Reset the state of the environment to an initial state
    cartPosIdeal = np.array([tableWidth/10, tableHeight/2])
    cartPosNoisy = np.random.normal(cartPosIdeal, 0.1)

    initialState = np.array([])

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass
  def reset(self):
    pass