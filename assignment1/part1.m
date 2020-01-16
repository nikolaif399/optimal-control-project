function [r, p, y] = part1( target, link_length, min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw, obstacles )
%% Function that uses optimization to do inverse kinematics for a snake robot

%%Outputs 
  % [r, p, y] = roll, pitch, yaw vectors of the N joint angles
  %            (N link coordinate frames)
%%Inputs:
    % target: [x, y, z, q0, q1, q2, q3]' position and orientation of the end
    %    effector
    % link_length : Nx1 vectors of the lengths of the links
    % min_xxx, max_xxx are the vectors of the 
    %    limits on the roll, pitch, yaw of each link.
    % limits for a joint could be something like [-pi, pi]
    % obstacles: A Mx4 matrix where each row is [ x y z radius ] of a sphere
    %    obstacle. M obstacles.

% Your code goes here.

N = length(link_length);

rpy0 = rand(N, 3);
    
%% Setup Optimization
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
problem.options = options;
problem.solver = 'fmincon';
problem.objective = @ (x) objective_fn(x,target,link_length,...
                                       min_roll, max_roll,...
                                       min_pitch, max_pitch,...
                                       min_yaw, max_yaw);
%problem.nonlcon = @(x) ;
problem.lb = [min_roll, min_pitch, min_yaw];
problem.ub = [max_roll, max_pitch, max_yaw];
problem.x0 = rpy0;
rpy = fmincon(problem);

r = rpy(:,1);
p = rpy(:,2);
y = rpy(:,3);

end