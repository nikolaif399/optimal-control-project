% State = [x,y,theta,phi,v,w] % phi is steering angle
% Action = [f, dphi]


%min(sum(dts))
% constrain dynamics
% constrain initial and final positions

% optimize state, t series and u
%% Setup and Solve Optimization
problem.options = optimoptions('fmincon','Display', 'iter', 'MaxFunctionEvaluations', 1e7, 'MaxIterations', 1e5);
problem.objective = @(x) objective_fn(x, N);
problem.nonlcon = @(x) nonlin_constraint_fn(x, N, nx, ndx, nu);

%problem.x0 = 0;
%problem.Aeq = Aeq;
%problem.beq = beq;
%problem.Aineq = Aineq;
%problem.Bineq = Bineq;

xout = fmincon(problem);