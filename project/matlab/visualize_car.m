function [] = visualize_car(x,y,theta,phi,car_params)
% visualize_car Visualize a simple car given vehicle center position,
% vehicle orientation, steering angle and car size (stored in params)
L = car_params.l;
W = car_params.w;

R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

%% ll lr ur ul
x_base = [-W/2 W/2 W/2 -W/2];
y_base = [-L/2 -L/2 L/2 L/2];

pos = zeros(5,2);
for i = 1:4
    pos(i,:) = R * [x_base(i); y_base(i)];
end
pos(5,:) = pos(1,:);

figure(1)
plot(pos(:,1), pos(:,2))
title("Car")
xlim([-3,3])
axis equal

end