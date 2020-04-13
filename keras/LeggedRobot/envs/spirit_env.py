import gym
from gym import spaces
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pybullet_data

class SpiritEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, render=False, _use_pos_control = False):
    super(SpiritEnv, self).__init__()

    # Setup spirit model

    # Left Front, Left Rear, Right Front, Right Rear
    self.abd_ids = [0,4,8,12]
    self.hip_ids = [1,5,9,13]
    self.knee_ids = [2,6,10,14]

    self.joint_ids = self.abd_ids + self.hip_ids + self.knee_ids
    self.num_joints = len(self.joint_ids)

    self.leg_dirs = np.array([1,1,-1,-1])

    self.knee_nom_pos = 1.2
    self.hip_nom_pos = np.pi/4
    self.abd_nom_pos = 0

    self.use_pos_control = _use_pos_control

    # Setup PyBullet Environment for dynamics simulation
    clientId = pb.GUI if render else pb.DIRECT
    self.physicsClient = bc.BulletClient(connection_mode=clientId)
    self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    self.tstep = 0.01

    # Setup gym environment specifics
    self.num_acts = 8
    self.appliedTorques = np.zeros((self.num_acts,1))
    self.min_torque = -10*np.ones((self.num_acts,1))
    self.max_torque = 10*np.ones((self.num_acts,1))

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


  def action2Torques(self, action):
    # Action scaled from ~ -1 to 1 (tanh output layer, random noise attached for exploration)
    action_column = action.reshape(self.num_acts,1)
    action_clipped = np.clip(action_column, -1, 1)

    scaling_factor = (self.max_torque-self.min_torque)/2
    return self.min_torque + np.multiply(1+action_clipped, scaling_factor)

  def step(self, action):
    # Execute one time step within the environment
    
    if (self.use_pos_control):
      self.physicsClient.setJointMotorControlArray(self.robotId,
                                 self.hip_ids,
                                 pb.POSITION_CONTROL,
                                 targetPositions=action[0:4,:])

      self.physicsClient.setJointMotorControlArray(self.robotId,
                                 self.knee_ids,
                                 pb.POSITION_CONTROL,
                                 targetPositions=action[4:8,:])
    
    else:
      self.appliedTorques = self.action2Torques(action)
      hipTorques = self.appliedTorques[0:4,:]
      kneeTorques = self.appliedTorques[4:8,:]
      
      self.physicsClient.setJointMotorControlArray(self.robotId,
                                   self.hip_ids,
                                   pb.TORQUE_CONTROL,
                                   forces=hipTorques)
      self.physicsClient.setJointMotorControlArray(self.robotId,
                                  self.knee_ids,
                                  pb.TORQUE_CONTROL,
                                  forces=kneeTorques)

    # Fixed abduction joint
    self.physicsClient.setJointMotorControlArray(self.robotId,
                                   self.abd_ids,
                                   pb.POSITION_CONTROL,
                                   targetPositions=self.abd_nom_pos*self.leg_dirs)

  
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
    self.physicsClient.setTimeStep(self.tstep)

    self.resetRobot()
    self.planeId = self.physicsClient.loadURDF("plane.urdf")

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
    joint_angles = [self.physicsClient.getJointState(self.robotId, joint_id)[0] for joint_id in self.joint_ids]
    return np.array(joint_angles).reshape(self.num_joints,1)

  def getJointVelocities(self):
    joint_velocities = [self.physicsClient.getJointState(self.robotId, joint_id)[1] for joint_id in self.joint_ids]
    return np.array(joint_velocities).reshape(self.num_joints,1)

  def getJointTorques(self):
    joint_torques = [self.physicsClient.getJointState(self.robotId, joint_id)[3] for joint_id in self.joint_ids]
    return np.array(joint_torques).reshape(self.num_joints,1)

  def computeObservation(self):
    linPos = self.getBasePosition()
    linVel = self.getBaseLinVel()
    angPos = self.getBaseOrientation()
    angVel = self.getBaseAngVel()
    jointPos = self.getJointPositions()
    jointVel = self.getJointVelocities()

    observation = np.vstack((linPos, linVel, angPos, angVel, jointPos, jointVel, self.appliedTorques.reshape(self.num_acts,1)))
    
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


  def resetJoint(self, joint_id, pos, addFTSensor=False):
    self.physicsClient.resetJointState(self.robotId,
                                           joint_id,
                                           pos,
                                           targetVelocity=0
                                           )
    self.physicsClient.setJointMotorControl2(self.robotId,
                                         joint_id,
                                         pb.VELOCITY_CONTROL,
                                         force=0)

    if (addFTSensor):
      self.physicsClient.enableJointForceTorqueSensor(self.robotId, joint_id)



  def resetRobot(self):
    robotStartPos = [0,0,0.3]
    robotStartOrn = self.physicsClient.getQuaternionFromEuler([0,0,0])
    self.robotId = self.physicsClient.loadURDF("assets/spirit40.urdf", robotStartPos, robotStartOrn)

    use_torque_sensor = self.use_pos_control

    for i in range(len(self.abd_ids)):
      abd_id = self.abd_ids[i]
      self.resetJoint(abd_id, self.leg_dirs[i]*self.abd_nom_pos, use_torque_sensor)

      hip_id = self.hip_ids[i]
      self.resetJoint(hip_id, self.hip_nom_pos, use_torque_sensor)

      knee_id = self.knee_ids[i]
      self.resetJoint(knee_id, self.knee_nom_pos, use_torque_sensor)