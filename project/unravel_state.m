function [x, y, theta, phi, v, w, dt] = unravel_state(q)
%UNRAVEL_STATE Reduces our complex representation of state vector

%q = [x y theta phi v w t x y ...];

x = q(1:7:end);
y = q(2:7:end);
theta = q(3:7:end);
phi = q(4:7:end);
v = q(5:7:end);
w = q(6:7:end);
dt = q(7:7:end);

end

