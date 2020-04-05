function critic = build_critic(obsInfo, actInfo)
%build_critic Setups RL critic
    numObs = obsInfo.Dimension(1);
    numAct = actInfo.Dimension(2);
    statePath = [
        imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
        fullyConnectedLayer(400,'Name','CriticStateFC1')
        reluLayer('Name', 'CriticRelu1')
        fullyConnectedLayer(300,'Name','CriticStateFC2')];
    actionPath = [
        imageInputLayer([numAct 1 1],'Normalization','none','Name','action')
        fullyConnectedLayer(300,'Name','CriticActionFC1','BiasLearnRateFactor',0)];
    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','CriticCommonRelu')
        fullyConnectedLayer(1,'Name','CriticOutput')];

    criticNetwork = layerGraph();
    criticNetwork = addLayers(criticNetwork,statePath);
    criticNetwork = addLayers(criticNetwork,actionPath);
    criticNetwork = addLayers(criticNetwork,commonPath);

    criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
    criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

    criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
    critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOpts);
end

