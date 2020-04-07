import gym
from gym import spaces
import numpy as np

class AirHockeyEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(AirHockeyEnv, self).__init__()
    
    # Define Air Hockey Physical Environment
    self.massStriker = 1.0
    self.massPuck = 0.5
    self.radiusStriker = 0.08
    self.radiusPuck = 0.06
    self.maxForce = 30 # Maximum force applied to striker

    self.tableWidth = 2 # X Dimension
    self.tableHeight = 1 # Y Dimension
    #frictionCoeff = 0.0 # Friction between puck and surface

    # Define action and observation space
    # They must be gym.spaces objects
    self.action_space = spaces.Box(low=-self.maxForce, high=self.maxForce, shape = (2,1), dtype=np.float32)

    self.min_pos = np.array([0,0]).reshape((2,1))
    self.max_pos = np.array([self.tableWidth, self.tableHeight]).reshape((2,1))
    self.max_striker_vel = np.array([3,3]).reshape((2,1))
    self.max_puck_vel = np.array([6,6]).reshape((2,1))
    self.min_obs = np.vstack((self.min_pos, -self.max_striker_vel, self.min_pos, -self.max_puck_vel))
    self.max_obs = np.vstack((self.max_pos, self.max_striker_vel, self.max_pos, self.max_puck_vel))

    self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, dtype=np.float32)

    self.state = None

  def step(self, action):
    # Execute one time step within the environment
    pass
    

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print(self.state)
    print(self.state.shape)

  def reset(self):
    # Reset the state of the environment to an initial state
    strikerPosIdeal = np.array([self.tableWidth/10, self.tableHeight/2]).reshape(2,1)
    strikerPos = np.random.normal(strikerPosIdeal, 0.1).reshape(2,1)

    strikerVel = np.random.normal(0, 0.3, (2,1)).reshape(2,1)

    puckPosIdeal = np.array([self.tableWidth/2, self.tableHeight/2]).reshape(2,1)
    puckPos = np.random.normal(puckPosIdeal, 0.2).reshape(2,1)

    puckVel = np.random.normal(0, 1, (2,1)).reshape(2,1)

    initialState = np.vstack((strikerPos, strikerVel, puckPos, puckVel))

    self.state = np.clip(initialState, self.min_obs, self.max_obs)
    return self.state








