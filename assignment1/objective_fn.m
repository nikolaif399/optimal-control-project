function cost = objective_fn(x, target, link_length,min_roll, max_roll,...
                                                    min_pitch, max_pitch,...
                                                    min_yaw, max_yaw)
%% objective_fn Determines the cost of any particular state

%% Relative weightings
pos_cost = 10; % meters
ang_cost = 100; % radians
Q_state = diag([pos_cost, pos_cost, pos_cost, ang_cost, ang_cost, ang_cost]);

cons_cost = 0.001; % radians
Q_constraint = cons_cost * eye(2*3*length(link_length));

%% Cost from position error

roll = x(:,1); pitch = x(:,2); yaw = x(:,3);
[~,pos_curr,~,ang_curr] = fk(roll, pitch, yaw, link_length);

pos_targ = target(1:3);
ang_targ = quat2eul(target(4:7)');

goal_error = [pos_targ - pos_curr; (ang_targ - ang_curr)'];
goal_cost = goal_error'*Q_state*goal_error;

%% Cost from proximity to constraints
roll_min_dist = abs(roll - min_roll);
roll_max_dist = abs(roll - max_roll);

pitch_min_dist = abs(pitch - min_pitch);
pitch_max_dist = abs(pitch - max_pitch);

yaw_min_dist = abs(yaw - min_yaw);
yaw_max_dist = abs(yaw - max_yaw);

constraint_dist = [roll_min_dist; roll_max_dist; pitch_min_dist;...
                   pitch_max_dist; yaw_min_dist; yaw_max_dist];
               
constraint_reward = constraint_dist'*Q_constraint*constraint_dist;

cost = goal_cost - constraint_reward;

end

