syms a b y real

R = [cos(a)*cos(b) cos(a)*sin(b)*sin(y)-sin(a)*cos(y) cos(a)*sin(b)*cos(y)+sin(a)*sin(y);
     sin(a)*cos(b) sin(a)*sin(b)*sin(y)+cos(a)*cos(y) sin(a)*sin(b)*cos(y)-cos(a)*sin(y);
     -sin(b) cos(b)*sin(y) cos(b)*cos(y)];
 
rpy = rand(1,3);

R1 = double(subs(R, [a b y], rpy))
R2 = eul2rotm(rpy)