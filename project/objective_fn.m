function cost = objective_fn(q, opt_params)
%objective_fn Defines the cost of the nonlinear system
    state = unravel_state(q, opt_params);
    cost = sum(state.dt) + state.a'*state.a;
end

