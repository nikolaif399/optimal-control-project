function state = unravel_state(q, params)
%UNRAVEL_STATE Reduces our complex representation of state vector

% q = [x y theta phi v dx dy dtheta dt];
% u = [a w];
% T = [dt];

% Recover state
state.x = q(1 : params.N);
state.y = q(params.N+1 : 2*params.N);
state.theta = q(2*params.N+1 : 3*params.N);
state.phi = q(3*params.N+1 : 4*params.N);
state.v = q(4*params.N+1 : 5*params.N);

% Recover control
state.a = q(5*params.N+1 : 6*params.N);
state.w = q(6*params.N+1 : 7*params.N);

% Recover time
state.dt = q(7*params.N+1 : 8*params.N - 1);

end

