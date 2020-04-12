import gym
from gym import spaces
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pybullet_data

class LeggedRobotEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, render=False):
    super(LeggedRobotEnv, self).__init__()

    self.num_joints = 12

    # Setup PyBullet Environment for dynamics simulation
    clientId = pb.GUI if render else pb.DIRECT
    self.physicsClient = bc.BulletClient(connection_mode=clientId)#pb.connect(clientId)
    self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

    self.jointIds = np.array([0,1,2,3,4,5,6,7,8,9,10,11]) # Figure out what the correct motor IDS are
    self.appliedTorques = np.zeros((self.num_joints,1))

    # Setup gym environment specifics
    self.min_torque = -10*np.ones((self.num_joints,1))
    self.max_torque = 10*np.ones((self.num_joints,1))

    self.action_space = spaces.Box(low=self.min_torque, high=self.max_torque, dtype=np.float64)
    
    min_pos = np.array([-np.inf, -np.inf, 0]).reshape((3,1))
    min_vel = np.array([-np.inf, -np.inf, -np.inf]).reshape((3,1))
    min_ori = np.array([-np.pi, -np.pi, -np.inf]).reshape((3,1))
    min_ang_vel = np.array([-np.inf, -np.inf, -np.inf]).reshape((3,1))
    min_joint_ang = -np.pi * np.ones((self.num_joints,1))
    min_joint_vel = -np.inf * np.ones((self.num_joints,1))

    max_pos = np.array([np.inf, np.inf, np.inf]).reshape((3,1))
    max_vel = np.array([np.inf, np.inf, np.inf]).reshape((3,1))
    max_ori = np.array([np.pi, np.pi, np.inf]).reshape((3,1))
    max_ang_vel = np.array([np.inf, np.inf, np.inf]).reshape((3,1))
    max_joint_ang = np.pi * np.ones((self.num_joints,1))
    max_joint_vel = np.inf * np.ones((self.num_joints,1))

    self.min_obs = np.vstack((min_pos, min_vel, min_ori, min_ang_vel, min_joint_ang, min_joint_vel, self.min_torque))
    self.max_obs = np.vstack((max_pos, max_vel, max_ori, max_ang_vel, max_joint_ang, max_joint_vel, self.max_torque))
    self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, dtype=np.float64)
    self.observation = []
    
    self.stepCount = 0


  def scaleAction(self, action):
    # Action scaled from -1 to 1 (tanh output layer)

    scaling_factor = (self.max_torque-self.min_torque)/2
    return self.min_torque + np.multiply(1+action.reshape(self.num_joints,1), scaling_factor )

  def step(self, action):
    # Execute one time step within the environment
    print(self.computeObservation())
    print(action)
    
    self.appliedTorques = self.scaleAction(action)
    

    self.physicsClient.setJointMotorControlArray(self.robotId,
                                 self.jointIds,
                                 pb.TORQUE_CONTROL,
                                 forces=self.appliedTorques)

    self.physicsClient.stepSimulation()
    self.stepCount += 1

    observation = self.computeObservation()
    reward = self.computeReward()
    done = self.computeDone()
    debug_info = {}
    return observation, reward, done, debug_info

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass

  def reset(self):
    # Reset the state of the environment to an initial state
    
    self.physicsClient.resetSimulation()
    self.physicsClient.setGravity(0,0,-9.81)
    self.physicsClient.setTimeStep(0.01)

    robotStartPos = [0,0,0.2]
    robotStartOrn = self.physicsClient.getQuaternionFromEuler([0,0,0])
    self.robotId = self.physicsClient.loadURDF("models/spirit40.urdf", robotStartPos, robotStartOrn)

    self.planeId = self.physicsClient.loadURDF("plane.urdf")

    for joint_id in self.jointIds:
        self.physicsClient.resetJointState(self.robotId,
                                           joint_id,
                                           0,
                                           targetVelocity=0
                                          )

    return self.computeObservation()

  def getBasePosition(self):
    robotPos,_ = self.physicsClient.getBasePositionAndOrientation(self.robotId)
    return np.array(robotPos).reshape(3,1)

  def getBaseLinVel(self):
    robotLinVel, _ = self.physicsClient.getBaseVelocity(self.robotId)
    return np.array(robotLinVel).reshape(3,1)

  def getBaseOrientation(self):
    _, robotAngQuat = self.physicsClient.getBasePositionAndOrientation(self.robotId)
    robotAng = self.physicsClient.getEulerFromQuaternion(robotAngQuat)
    return np.array(robotAng).reshape(3,1)

  def getBaseAngVel(self):
    _, robotAngVel = self.physicsClient.getBaseVelocity(self.robotId)
    return np.array(robotAngVel).reshape(3,1)

  def getJointPositions(self):
    joint_angles = [self.physicsClient.getJointState(self.robotId, jointId)[0] for jointId in self.jointIds]
    return np.array(joint_angles).reshape(self.num_joints,1)

  def getJointVelocities(self):
    joint_velocities = [self.physicsClient.getJointState(self.robotId, jointId)[1] for jointId in self.jointIds]
    return np.array(joint_velocities).reshape(self.num_joints,1)

  def computeObservation(self):
    linPos = self.getBasePosition()
    linVel = self.getBaseLinVel()
    angPos = self.getBaseOrientation()
    angVel = self.getBaseAngVel()
    jointPos = self.getJointPositions()
    jointVel = self.getJointVelocities()

    observation = np.vstack((linPos, linVel, angPos, angVel, jointPos, jointVel, self.appliedTorques.reshape(12,1)))
    
    return observation

  def computeReward(self):
    linPos = self.getBasePosition()
    linVel = self.getBaseLinVel()
    angPos = self.getBaseOrientation()
    angVel = self.getBaseAngVel()

    xVel = linVel[0] # Robot Forward Velocity
    roll = angPos[0]
    pitch = angPos[1]
    torques = self.appliedTorques

    reward_arr = 100*xVel - 1*(roll**2 + pitch**2) - 0.01*np.sum(torques**2)
    reward = reward_arr.item()

    return reward
  
  def computeDone(self):
    linPos = self.getBasePosition()

    zPos = linPos[2]

    return zPos < 0.15 or self.stepCount > 500


