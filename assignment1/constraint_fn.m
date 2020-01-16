function [c, ceq] = constraint_fn(x, obstacles, link_length,min_roll, max_roll,...
                                                    min_pitch, max_pitch,...
                                                    min_yaw, max_yaw)
%% constraint_fn Ensures arm does not pass through obstacles
c = [];
ceq = [];

%% For each link, determine the minimum distance to the center of each
% obstacle. Ensure every possible one of these distances is greater than
% the radius of the relevant obstacle

N = length(link_length);
M = size(obstacles,1);

last_frame_pos = zeros(3,1);
for i = 1:N
    [~,curr_frame_pos] = fk(r,p,y,link_length,i);
    
    for j = 1:M
        x1 = last_frame_pos;
        x2 = curr_frame_pos;
        x0 = obstacles(j,1:3);
        
        % http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        t = -dot(x1 - x0,x2-x1)/(norm(x2-x1)^2);
        if (t < 0 || t > 1)
            % closest point along line is not on line segment, use closest
            % endpoint instead
            d = norm(cross(x0-x1,x0-x2))/norm(x2-x1);
        else
            d = 0; % something else here
        end
        % d = minimum distance from link i to obstacle j
        
        % impose this as a constraint, must be greater than radius
    end
    
    last_frame_pos = curr_frame_pos;
end

end

