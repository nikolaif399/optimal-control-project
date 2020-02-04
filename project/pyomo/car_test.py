#!/usr/bin/env python

from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import time

# Shorthand indices for various state quantities
_X = 1
_Y = 2
_TH = 3
_PH = 4
_V = 5

_A = 1
_W = 2

# Car parameters (constants)
L = 2.5


# Direct Collocation parameters
nx = 5
nu = 2
N = 1000

T = 10
dt = T/N

# Obstacle parameters
num_obs = 2

num_obs = 3

m = ConcreteModel()

# Store indexing in model
m.states = RangeSet(1,nx)  # i
m.controls = RangeSet(1,nu)  # j

m.kx = RangeSet(0, N)  # k
m.ku = RangeSet(0, N-1)
m.ko = RangeSet(1, num_obs)


# Parameters
m.xmin = Param(m.states, initialize = {_X:-100, _Y:-100, _TH:-4*np.pi, _PH:-2*np.pi/3, _V:-10})
m.xmax = Param(m.states, initialize = {_X: 100, _Y: 100, _TH: 4*np.pi, _PH: 2*np.pi/3, _V: 10})

m.umin = Param(m.controls, initialize={_A:-5, _W:-5})
m.umax = Param(m.controls, initialize={_A: 5, _W: 5})

m.pprint()
def init_obstacles(m, i):
  print(i)
  if (i == 1): return {1:2, 3:4}
  if (i == 2): return {3:5, 2:4}
  if (i == 3): return {1:2, 3:4}

m.obs = Param(m.ko, initialize=init_obstacles)

m.ic = Param(m.states, initialize=0)
m.fc = Param(m.states, initialize={_X:10, _Y:10, _TH:np.pi/4, _PH:0, _V:0})


def init_obstacles(m, i, j):
  if (j == 1): return [3,4,1][i]
  if (j == 2): return [6,2,1][i]
  if (j == 3): return [3,2,1][i]
  if (j == 4): return [3,0,1][i]


m.o = Param(m.obstacles, m.ko, initialize=init_obstacles)

# State Variables
m.x = Var(m.states, m.kx, bounds = lambda m,i,j: (m.xmin[i], m.xmax[i]), initialize = 0)
m.icfix = Constraint(m.states, rule = lambda m,i: m.x[i,0] == m.ic[i])
m.fcfix = Constraint(m.states, rule = lambda m,i: m.x[i,N] == m.fc[i])

# Control Variables
m.u = Var(m.controls, m.ku, bounds=lambda m,i,j: (m.umin[i], m.umax[i]), initialize=0)
m.wu = {_A:5, _W:10}

# State update constraints
def x_update_fn(m,i,j):
  if i == N: return Constraint.Skip

  if  (j == _X): return (m.x[_X, i+1] == m.x[_X, i] + m.x[_V,i]*cos(m.x[_TH,i]) * dt)
  if  (j == _Y): return (m.x[_Y, i+1] == m.x[_Y, i] + m.x[_V,i]*sin(m.x[_TH,i]) * dt)
  if (j == _TH): return (m.x[_TH,i+1] == m.x[_TH,i] + m.x[_V,i]*tan(m.x[_PH,i])/L * dt) 
  if (j == _PH): return (m.x[_PH,i+1] == m.x[_PH,i] + m.u[_W,i]*dt)
  if (j == _V): return  (m.x[_V, i+1] == m.x[_V, i] + m.u[_A,i]*dt) 

m.x_update = Constraint(m.kx, m.states, rule=x_update_fn)

# Path constraints
def path_constraint_fn(m,i,j):
  obsx, obsy, obsr2 = m.o[0,j], m.o[1,j], m.o[2,j]
  xerror = (obsx - m.x[_X,i])
  yerror = (obsy - m.x[_Y,i])
  return (pow(xerror, 2) + pow(yerror, 2) >= obsr2)

m.path_constraint = Constraint(m.kx, m.ko, rule=path_constraint_fn)
m.path_constraint.deactivate()

# Objective function
def obj_min_acc_fn(m):
  res = 0
  for j in range(1,nu+1):
    for k in range(0,N):
      res += m.wu[j]*pow(m.u[j,k],2)
  return res

m.obj = Objective(rule=obj_min_acc_fn)

# Solve
opt = SolverFactory('ipopt')

mid_time = time.time()
results = opt.solve(m)
end_time = time.time()
m.path_constraint.activate()
results = opt.solve(m)
end2_time = time.time()

setup_time = mid_time - start_time
exec_time = end_time - mid_time
exec2_time = end2_time - mid_time
print("Setup time: %f s" % setup_time)
print("Solve time: %f s" % exec_time)
print("Solve2 time: %f s" % exec2_time)

# Graph x and u
x = [value(m.x[_X,k]) for k in range(0,N+1)]
y = [value(m.x[_Y,k]) for k in range(0,N+1)]
theta = [value(m.x[_TH,k]) for k in range(0,N+1)]
phi = [value(m.x[_PH,k]) for k in range(0,N+1)]
v = [value(m.x[_V,k]) for k in range(0,N+1)]

a = [value(m.u[_A,k]) for k in range(0,N)]
w = [value(m.u[_W,k]) for k in range(0,N)]

# Graph obstacles in X,Y
xobs = [value(m.o[0,i]) for i in range(1,num_obs+1)]
yobs = [value(m.o[1,i]) for i in range(1,num_obs+1)]
robs = [value(m.o[2,i]) for i in range(1,num_obs+1)]

t = np.linspace(0, T, N+1)

fig, axes = plt.subplots( 3, 3 )

axes[0,0].set_title('X')
axes[0,0].plot(t,x)

axes[0,1].set_title('y')
axes[0,1].plot(t,y)

axes[0,2].plot(t,theta)
axes[0,2].set_title('Theta')

axes[1,0].plot(t,phi)
axes[1,0].set_title('Phi')

axes[1,1].plot(t,v)
axes[1,1].set_title('V')

axes[1,2].plot(t[:-1],w)
axes[1,2].set_title('Omega')

axes[2,0].plot(t[:-1],a)
axes[2,0].set_title('Acceleration')

axes[2,1].plot(x,y)
axes[2,1].set_title('Path  (x vs y)')

for i in range(1,num_obs+1):
  c = plt.Circle((xobs[i-1],yobs[i-1]), robs[i-1], color='r')
  axes[2,1].add_patch(c)

plt.show()
