%% Forward Kinematics
r = rand(6,1);
p = rand(6,1);
y = rand(6,1);
l = rand(6,1);
[T1, p1, q1] = fk(r,p,y,l);
[T2, p2, q2] = fk(r,p,y,l, 4);

%% Arm Setup
min_roll = -pi*ones(3,1);
max_roll = pi*ones(3,1);
min_pitch = -pi*ones(3,1);
max_pitch = pi*ones(3,1);
min_yaw = -pi*ones(3,1);
max_yaw = pi*ones(3,1);
link_length = [0.2;0.2;0.2];

obstacles = [0.5 0 0 0.05];

%% Part 1
pos_targ = [0.4; 0; 0];
ang_targ = eul2quat([0 0.5 0])';

target = [pos_targ;ang_targ];
[r,p,y] = part1( target, link_length, min_roll, max_roll,...
                                        min_pitch, max_pitch,...
                                        min_yaw, max_yaw, obstacles);
                                  
visualize(r,p,y,link_length,obstacles);                               
%[T,pos,quat,eul] = fk(r,p,y,link_length);

