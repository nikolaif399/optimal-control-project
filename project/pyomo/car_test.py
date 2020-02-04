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

start_time = time.time()
nx = 5
nu = 2
N = 1000

T = 10
dt = T/N

obsx = 5
obsy = 2
obsr = 3
obsr2 = pow(obsr, 2)

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
m.fc = Param(m.states, initialize={_X:10, _Y:10, _TH:np.pi/2, _PH:0, _V:0})

# State Variables
m.x = Var(m.states, m.kx, bounds = lambda m,i,k: (m.xmin[i], m.xmax[i]), initialize = 0)
m.icfix = Constraint(m.states, rule = lambda m,i: m.x[i,0] == m.ic[i])
m.fcfix = Constraint(m.states, rule = lambda m,i: m.x[i,N] == m.fc[i])

# Control Variables
m.u = Var(m.controls, m.ku, bounds=lambda m,j,k: (m.umin[j], m.umax[j]), initialize=0)
m.wu = {_A:5, _W:10}

# State update constraints
def x_update_fn(m,k,i):
  if k == N: return Constraint.Skip

  if  (i == _X): return (m.x[_X, k+1] == m.x[_X, k] + m.x[_V,k]*cos(m.x[_TH,k]) * dt)
  if  (i == _Y): return (m.x[_Y, k+1] == m.x[_Y, k] + m.x[_V,k]*sin(m.x[_TH,k]) * dt)
  if (i == _TH): return (m.x[_TH,k+1] == m.x[_TH,k] + m.x[_V,k]*tan(m.x[_PH,k])/L * dt)
  if (i == _PH): return (m.x[_PH,k+1] == m.x[_PH,k] + m.u[_W,k]*dt)
  if (i == _V): return  (m.x[_V, k+1] == m.x[_V, k] + m.u[_A,k]*dt) 

m.x_update = Constraint(m.kx, m.states, rule=x_update_fn)

# Path constraints
def path_constraint_fn(m,k):
  xerror = (obsx - m.x[_X,k])
  yerror = (obsy - m.x[_Y,k])
  return (pow(xerror, 2) + pow(yerror, 2) >= obsr2)

m.path_constraint = Constraint(m.kx, rule=path_constraint_fn)
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

t = np.linspace(0, T, N+1)

plt.subplot(3,3,1)
plt.title('X')
plt.plot(t,x)

plt.subplot(3,3,2)
plt.title('y')
plt.plot(t,y)

plt.subplot(3,3,3)
plt.plot(t,theta)
plt.title('Theta')

plt.subplot(3,3,4)
plt.plot(t,phi)
plt.title('Phi')

plt.subplot(3,3,5)
plt.plot(t,v)
plt.title('V')

plt.subplot(3,3,6)
plt.plot(t[:-1],w)
plt.title('Omega')

plt.subplot(3,3,7)
plt.plot(t[:-1],a)
plt.title('Acceleration')

plt.subplot(3,3,8)
plt.plot(x,y)
plt.title('Path  (x vs y)')


plt.show()