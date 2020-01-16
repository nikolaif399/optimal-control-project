function [T,pos,quat,eul] = fk(roll, pitch, yaw, link_length, link_num)
%% Function that computes the end effector position and orientation

%% Outputs:
    % T = position and orientation of the end of the specified link (mat)

%% Inputs 
  % [r, p, y] = roll, pitch, yaw vectors of the N joint angles
  %            (N link coordinate frames)
  % link_length : Nx1 vectors of the lengths of the links
  % link_num : which link to get transform to

% Your code goes here.
assert(length(roll) == length(pitch));
assert(length(roll) == length(yaw));

N = length(roll);
if (nargin > 4)
    N = min(link_num, N);
end

T = eye(4);
for i = 1:N
    % Rotation
    rpy = [roll(i) pitch(i) yaw(i)];
    Ri = eul2rotm(rpy);
    
    T = T*[Ri [0;0;0]; 0 0 0 1];
    
    % Translation
    p = [link_length(i); 0; 0];
    Ti = [eye(3) p; 0 0 0 1];
    
    T = T * Ti;
end

pos = T(1:3,4);
quat = rotm2quat(T(1:3,1:3));
eul = rotm2eul(T(1:3,1:3));
end