classdef leggedRobotEnv < rl.env.MATLABEnvironment
    %LEGGEDROBOTENV: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties   
        
        % Acceleration due to gravity in m/s^2
        Gravity = 9.81
            
        % Sample time
        Ts = 0.01
        
        % Robot Physical Params
        L_body = 0.4;
        H_body = 0.1;
        
        L_link1 = 0.1;
        L_link2 = 0.2;
        
        % Dynamic Matrices
        m_body = 10;
        i_body = 1/12*m_body*(l_body^2 + l_body^2);
        M_body = diag([m_body, m_body, i_body]);
        J_body = 0;
        
        
        m_link1 = 0.5;
        i_link1 = 1/12*m_link1*l_link1^2;
        M_link1 = diag([m_link1, m_link1, i_link1]);
        J_link1_front = 0;
        J_link1_back = 0;
        
        m_link2 = 0.3;
        i_link2 = 1/12*m_link2*l_link2^2;
        M_link2 = diag([m_link2, m_link2, i_link2]);
        J_link2_front = 0;
        J_link2_back = 0;
        
        M = J_body' * M_body * J + ...
            J_link1_back' * M_link1 * J_link1_back + ...
            J_link2_back' * M_link2 * J_link2_back + ...
            J_link1_front' * M_link1 * J_link1_front + ...
            J_link2_front' * M_link2 * J_link2_front;
        
        q = sym('q', [7 1]);
        dq = sym('dq', [7 1]);
        C = get_coriolis_matrix(M,q,dq);
        
        % Reward each time step the cart-pole is balanced
        RewardForNotFalling = 1
        
        % Penalty when the cart-pole fails to balance
        PenaltyForFalling = -10 
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State = zeros(4,1)
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
        function this = leggedRobotEnv()
            
            % Initialize Observation settings
            numObs = 4;
            ObservationInfo = rlNumericSpec([numObs 1]);
            ObservationInfo.Name = 'CartPole States';
            ObservationInfo.Description = 'x, dx, theta, dtheta';
            
            % Initialize Action settings
            numAct = 1;
            ActionInfo = rlNumericSpec([numAct 1]);
            ActionInfo.Name = 'CartPole Action';
            
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
            % X 
            X0 = 0;
            % Xdot
            Xd0 = 0;
            % Theta (+- .05 rad)
            T0 = 0.2;%2 * 0.05 * rand - 0.05;  
            % Thetadot
            Td0 = 0;
            
            InitialObservation = [X0;Xd0;T0;Td0;];
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        % Discrete force 1 or 2
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
