function [] = visualize(r,p,y,link_length,obstacles)
%visualize Draw the arm and the obstacles in 3D space

X = 0; Y = 0; Z = 0; % Add origin point

N = length(link_length);
M = size(obstacles, 1);

for i = 1:N
    T = fk(r,p,y,link_length,i);
    frame_pos = T(1:3,4);
    X = [X frame_pos(1)];
    Y = [Y frame_pos(2)];
    Z = [Z frame_pos(3)];
end

figure(1)
hold on
title("Inverse Kinematics Result")
plot3(X,Y,Z, 'LineWidth', 1);
grid
[x,y,z] = sphere;
C = zeros(length(x),length(x),3);

for j = 1:M
    x0 = obstacles(j,1);
    y0 = obstacles(j,2);
    z0 = obstacles(j,3);
    r = obstacles(j,4);
    
    surf(x*r + x0, y*r + y0, z*r + z0, C)
end

axis equal

end

