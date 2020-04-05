simOpts = rlSimulationOptions('MaxSteps',200);

experience = sim(env,agent, simOpts);

experience.Observation.CartPoleStates.Data(1,:)