function actor = build_actor(obsInfo, actInfo)
%build_actor Setups RL Actor
    numObs = obsInfo.Dimension(1);
    numAct = actInfo.Dimension(2);
    actorNetwork = [
        imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
        fullyConnectedLayer(128,'Name','ActorFC1')
        reluLayer('Name','ActorRelu1')
        fullyConnectedLayer(200,'Name','ActorFC2')
        reluLayer('Name','ActorRelu2')
        fullyConnectedLayer(1,'Name','ActorFC3')
        tanhLayer('Name','ActorTanh1')
        scalingLayer('Name','ActorScaling','Scale',actInfo.UpperLimit)];

    actorOptions = rlRepresentationOptions('LearnRate',5e-04,'GradientThreshold',1);

    actor = rlRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling'},actorOptions);
end

