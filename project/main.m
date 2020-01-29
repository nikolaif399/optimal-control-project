


% optimize state, t series and u

%% Setup and Solve Optimization
car_params.L = 2;
opt_params.N = 20;
opt_params.Nx = 5;
opt_params.Nu = 2;
opt_params.Nt = 1;

q_len = opt_params.N*(opt_params.Nx + opt_params.Nu) +...
                  (opt_params.N-1)*opt_params.Nt;
N = opt_params.N;
              
% Initial and final conditions
Aeq = zeros(10, q_len);
Aeq(1,1) = 1;
Aeq(2,   N) = 1;
Aeq(3,   N+1) = 1;
Aeq(4, 2*N) = 1;
Aeq(5, 2*N+1) = 1;
Aeq(6, 3*N) = 1;
Aeq(7, 3*N+1) = 1;
Aeq(8, 4*N) = 1;
Aeq(9, 4*N+1) = 1;
Aeq(10,5*N) = 1;

x0 = 0;
y0 = 0;
theta0 = 0;
phi0 = 0;
v0 = 0;

xf = 2;
yf = 0;
thetaf = 0;
phif = 0;
vf = 0;

beq = [x0; xf; y0; yf; theta0; thetaf; phi0; phif; v0; vf];

% State Bounds
x_min = -10;       x_max = 10;
y_min = -10;       y_max = 10;
theta_min = -4*pi; theta_max = 4*pi;
phi_min = -pi/3;    phi_max = pi/3;
v_min = -10;       v_max = 10;
a_min = -2;        a_max = 10;
w_min = -pi;       w_max = pi;
dt_min = 0;        dt_max = 0.2;

problem.options = optimoptions('fmincon','Display', 'iter', 'MaxFunctionEvaluations', 1e7, 'MaxIterations', 1e5);
problem.objective = @(x) objective_fn(x, opt_params);
problem.nonlcon = @(x) nonlin_constraint_fn(x, opt_params, car_params);
problem.solver = 'fmincon';
problem.x0 = 0*rand(q_len, 1);

problem.lb = [x_min*ones(N,1); y_min*ones(N,1); theta_min*ones(N,1); phi_min*ones(N,1); v_min*ones(N,1); a_min*ones(N,1); w_min*ones(N,1); dt_min*ones(N-1,1); ];
problem.ub = [x_max*ones(N,1); y_max*ones(N,1); theta_max*ones(N,1); phi_max*ones(N,1); v_max*ones(N,1); a_max*ones(N,1); w_max*ones(N,1); dt_max*ones(N-1,1); ];
problem.Aeq = Aeq;
problem.beq = beq;

%problem.Aineq = Aineq;
%problem.Bineq = Bineq;

xout = fmincon(problem);

state = unravel_state(xout, opt_params);

tout = 0;
for i = 1:N-1
    tout = [tout; tout(end)+state.dt(i)];
end

figure(1)

subplot(2,3,1)
plot(tout,state.x)
title("X")

subplot(2,3,2)
plot(tout,state.y)
title("Y")

subplot(2,3,3)
plot(tout,state.theta)
title("Theta")

subplot(2,3,4)
plot(tout,state.phi)
title("Phi")

subplot(2,3,5)
plot(tout,state.v)
title("Velocity")
