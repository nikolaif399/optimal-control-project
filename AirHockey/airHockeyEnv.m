classdef airHockeyEnv < rl.env.MATLABEnvironment
    %AIRHOCKEYENV: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        
        FrictionCoeff = 0.0; % Between puck and surface
        
        PuckMass = 1.0;
        
        PuckRadius = 0.06;
        
        StrikerMass = 0.1;
        
        StrikerRadius = 0.08;
        
        % Max Force the input can apply (Fx and Fy)
        MaxForce = 10;
               
        % Sample time
        Ts = 0.02;
        
        RewardForScoring = 1;
        
        PenaltyForMissing = -10;
    end
    
    properties
        % Initialize system state [x_striker,  y_striker, x_puck,  y_puck,
        %                          dx_stiker, dy_striker, dx_puck, dy_puck]'
        State = zeros(8,1);
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
        
        % Handle to figure
        Figure
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = invertedPendulumEnv()
            
            
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([8 1]);
            ObservationInfo.Name = 'Air Hockey State';
            
            % Initialize Action settings
            ActionInfo = rlNumericSpec([2 1]);
            ActionInfo.Name = 'Striker Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            
            % Get action
            Force = getForce(this,Action);            
            
            % Unpack state vector
            XDot = this.State(2);
            Theta = this.State(3);
            ThetaDot = this.State(4);
            
            % Cache to avoid recomputation
            CosTheta = cos(Theta);
            SinTheta = sin(Theta);            
            SystemMass = this.CartMass + this.PoleMass;
            temp = (Force + this.PoleMass*this.HalfPoleLength * ThetaDot^2 * SinTheta) / SystemMass;

            % Apply motion equations            
            ThetaDotDot = (this.Gravity * SinTheta - CosTheta* temp) / (this.HalfPoleLength * (4.0/3.0 - this.PoleMass * CosTheta * CosTheta / SystemMass));
            XDotDot  = temp - this.PoleMass*this.HalfPoleLength * ThetaDotDot * CosTheta / SystemMass;
            
            % Euler integration
            Observation = this.State + this.Ts.*[XDot;XDotDot;ThetaDot;ThetaDotDot];

            % Update system states
            this.State = Observation;
            
            % Check terminal condition
            X = Observation(1);
            Theta = Observation(3);
            IsDone = abs(X) > this.DisplacementThreshold || abs(Theta) > this.AngleThreshold;
            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            
           
            
            InitialObservation = [X0_striker;  Y0_striker;  X0_puck;  Y0_puck;
                                  DX0_striker; DY0_striker; DX0_puck; DY0_puck];
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        function force = getForce(this,action)
            force = action;           
        end
        % update the action info based on max force
        function updateActionInfo(this)
            this.ActionInfo.LowerLimit = -this.MaxForce;
            this.ActionInfo.UpperLimit = this.MaxForce;
        end
        
        % Reward function
        function Reward = getReward(this)
            if ~this.IsDone
                Reward = this.RewardForNotFalling;
            else
                Reward = this.PenaltyForFalling;
            end          
        end
        
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            this.Figure = figure('Visible','on','HandleVisibility','off');
            
            ha = gca(this.Figure);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            
            ha.XLim = [-3 3];
            ha.YLim = [-1.5 1.5];
            
            hold(ha, 'on');
            
            % Update the visualization
            envUpdatedCallback(this)
        end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',4},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        function set.HalfPoleLength(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','HalfPoleLength');
            this.HalfPoleLength = val;
            notifyEnvUpdated(this);
        end
        function set.Gravity(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Gravity');
            this.Gravity = val;
        end
        function set.CartMass(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','CartMass');
            this.CartMass = val;
        end
        function set.PoleMass(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','PoleMass');
            this.PoleMass = val;
        end
        function set.MaxForce(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','MaxForce');
            this.MaxForce = val;
            updateActionInfo(this);
        end
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
        function set.AngleThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','AngleThreshold');
            this.AngleThreshold = val;
        end
        function set.DisplacementThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','DisplacementThreshold');
            this.DisplacementThreshold = val;
        end
        function set.RewardForNotFalling(this,val)
            validateattributes(val,{'numeric'},{'real','finite','scalar'},'','RewardForNotFalling');
            this.RewardForNotFalling = val;
        end
        function set.PenaltyForFalling(this,val)
            validateattributes(val,{'numeric'},{'real','finite','scalar'},'','PenaltyForFalling');
            this.PenaltyForFalling = val;
        end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            if ~isempty(this.Figure) && isvalid(this.Figure)
                % Set visualization figure as the current figure
                ha = gca(this.Figure);

                % Extract the cart position and pole angle
                x = this.State(1);
                theta = this.State(3);
                

                cartplot = findobj(ha,'Tag','cartplot');
                poleplot = findobj(ha,'Tag','poleplot');
                if isempty(cartplot) || ~isvalid(cartplot) ...
                        || isempty(poleplot) || ~isvalid(poleplot)
                    % Initialize the cart plot
                    cartpoly = polyshape([-0.25 -0.25 0.25 0.25],[-0.125 0.125 0.125 -0.125]);
                    cartpoly = translate(cartpoly,[x 0]);
                    cartplot = plot(ha,cartpoly,'FaceColor',[0.8500 0.3250 0.0980]);
                    cartplot.Tag = 'cartplot';

                    % Initialize the pole plot
                    L = this.HalfPoleLength*2;
                    polepoly = polyshape([-0.1 -0.1 0.1 0.1],[0 L L 0]);
                    polepoly = translate(polepoly,[x,0]);
                    polepoly = rotate(polepoly,rad2deg(-theta),[x,0]);
                    poleplot = plot(ha,polepoly,'FaceColor',[0 0.4470 0.7410]);
                    poleplot.Tag = 'poleplot';
                else
                    cartpoly = cartplot.Shape;
                    polepoly = poleplot.Shape;
                end

                % Compute the new cart and pole position
                [cartposx,~] = centroid(cartpoly);
                [poleposx,poleposy] = centroid(polepoly);
                dx = x - cartposx;
                dtheta = -theta - atan2(cartposx-poleposx,poleposy-0.25/2);
                cartpoly = translate(cartpoly,[dx,0]);
                polepoly = translate(polepoly,[dx,0]);
                polepoly = rotate(polepoly,rad2deg(dtheta),[x,0.25/2]);

                % Update the cart and pole positions on the plot
                cartplot.Shape = cartpoly;
                poleplot.Shape = polepoly;

                % Refresh rendering in the figure window
                drawnow();
            else
                plot(this);
            end
        end
    end
end
