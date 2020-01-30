#!/usr/bin/env python

from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import time


start_time = time.time()
nx = 2
nu = 1
N = 500

T = 5
dt = T/N

m = ConcreteModel()

# Store indexing in model
m.states = RangeSet(1,nx) # i
m.controls = RangeSet(1,nu) # j

m.kx = RangeSet(0,N) # k
m.ku = RangeSet(0,N-1)

# Parameters
m.xmin = Param(m.states, initialize = {1:-100, 2:-10})
m.xmax = Param(m.states, initialize = {1:100, 2:10})
m.ic = Param(m.states, initialize=0)
m.fc = Param(m.states, initialize={1:10, 2:-1})

# State Variables
m.x = Var(m.states, m.kx, bounds = lambda m,i,k: (m.xmin[i], m.xmax[i]), initialize = 0)
m.icfix = Constraint(m.states, rule = lambda m,i: m.x[i,0] == m.ic[i])
m.fcfix = Constraint(m.states, rule = lambda m,i: m.x[i,N] == m.fc[i])

# Control Variables
m.u = Var(m.controls, m.ku, bounds=(-10,10), initialize=0)

# State update constraint
def x1_update_fn(m,k):
  if k == N: return Constraint.Skip
  return (m.x[1,k+1] == m.x[1,k] + m.x[2,k]*dt)

m.x1_update = Constraint(m.kx, rule=x1_update_fn)

def x2_update_fn(m,k):
  if k == N: return Constraint.Skip
  return (m.x[2,k+1] == m.x[2,k] + m.u[1,k]*dt)

m.x2_update = Constraint(m.kx, rule=x2_update_fn)

# Objective function
def obj_min_acc_fn(m):
  res = 0
  for j in range(1,nu+1):
    for k in range(0,N):
      res += pow(m.u[j,k],2)
  return res

m.obj = Objective(rule=obj_min_acc_fn)

# Solve
opt = SolverFactory('ipopt')

mid_time = time.time()
results = opt.solve(m)
end_time = time.time()

setup_time = mid_time - start_time
exec_time = end_time - mid_time
print("Setup time: %f s" % setup_time)
print("Solve time: %f s" % exec_time)

# Display model
#m.pprint()

# Graph x and u

x1 = [value(m.x[1,k]) for k in range(0,N+1)]
x2 = [value(m.x[2,k]) for k in range(0,N+1)]

u = [value(m.u[1,k]) for k in range(0,N)]

t = np.linspace(0, T, N+1)

plt.subplot(3,1,1)
plt.title('Trajectory Optimization')
plt.plot(t,x1)
plt.ylabel('Position')

plt.subplot(3,1,2)
plt.plot(t,x2)
plt.ylabel('Velocity')

plt.subplot(3,1,3)
plt.plot(t[:-1],u)
plt.ylabel('Control')

plt.xlabel("Time")
plt.show()