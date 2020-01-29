function [c,ceq] = nonlin_constraint_fn(q, opt_params, car_params)
%nonlin_constraint_fn Defines the forward dynamics of the system
    c = [];
    ceq = zeros(opt_params.Nx*(opt_params.N-1),1);
    
    state = unravel_state(q,opt_params);
    for i = 1:opt_params.N-1
        % Active constraint index
        ind = (i-1)*opt_params.Nx + 1;
        
        % Mean values in interval
        v = (state.v(i) + state.v(i+1))/2;
        a = (state.a(i) + state.a(i+1))/2;
        w = (state.w(i) + state.w(i+1))/2;
        
        ceq(ind) = state.x(i) + v*cos(state.theta(i))*state.dt(i) - state.x(i+1);
        ceq(ind+1) = state.y(i) + v*sin(state.theta(i))*state.dt(i) - state.y(i+1);
        ceq(ind+2) = state.theta(i) + v*tan(state.phi(i))/car_params.L*state.dt(i) - state.theta(i+1);
        ceq(ind+3) = state.phi(i) + w*state.dt(i) - state.phi(i+1);
        ceq(ind+4) = state.v(i) + a*state.dt(i) - state.v(i+1);
    end  
end

