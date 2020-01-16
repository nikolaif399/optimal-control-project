function cost = objective_fn(x, target, link_length)
%% objective_fn Determines the cost of any particular state


%% Weight matrix
pos_cost = 1;
ang_cost = 10;
Q = diag([pos_cost, pos_cost, pos_cost, ang_cost, ang_cost, ang_cost]);

%% Current position
roll = x(:,1); pitch = x(:,2); yaw = x(:,3);
[~,pos_curr,~,ang_curr] = fk(roll, pitch, yaw, link_length);

%% Target position
pos_targ = target(1:3);
ang_targ = quat2eul(target(4:7)');

error = [pos_targ - pos_curr; (ang_targ - ang_curr)'];

cost = error'*Q*error;

end

