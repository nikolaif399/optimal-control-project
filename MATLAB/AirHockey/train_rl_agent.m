% Creating Environment
env = invertedPendulumEnv();
validateEnvironment(env)
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Creating RL Entities
actor = build_actor(obsInfo, actInfo);
critic = build_critic(obsInfo, actInfo);

agentOptions = rlDDPGAgentOptions();
agent = rlDDPGAgent(actor,critic,agentOptions);

% Training Agent in Environment
opt = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',1000,...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',480);

trainStats = train(agent,env,opt);