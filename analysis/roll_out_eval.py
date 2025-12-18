import numpy as np
import torch
import matplotlib.pyplot as plt

from models.mlp import MLP
from dynamics.pendulum import pendulum_dynamics
from dynamics.rk4_step import rk4_step
from dynamics.zero_control import zero_control
from simulation.rollout import simulate_trajectory
from analysis.mechanical_energy import mechanical_energy

# Load model
model = MLP()
model.load_state_dict(torch.load("mlp.pth"))
model.eval()

# Load data
data = np.load("data/pendulum_dataset.npz")

#--------------
# Parameters
#--------------
dt = data["dt"]
params = {"g":9.81, "l":1.0, "m":1.0}

#------------------
# Initialisation
#------------------
x0 = np.array([0.5,0.0])
T = 5.0

#------------------
# True trajectory
#------------------

true_traj = trajectory_simulation(x0, zero_control, dt, T, params)
u_traj = np.zeros(( int(T/dt), 1))

#-----------
# MLP traj
#-----------

x = x0.copy()
mlp_traj = [x]

for t in range(int(T/dt)):
  inp = torch.tensor(np.hstack([x,u_traj[t]]),
                     dtype=torch.float32
                     )
  with torch.no_grad():
    x = model(inp)
    mlp_traj.append(x)

mlp_traj = np.array(mlp_traj)

#-------
# Plot
#-------

t = np.arange(T/dt +1) * dt

# Trajectory
plt.plot(t, true_traj[:,0], label = "True")
plt.plot(t, mlp_traj[:,0],"--", label = "MLP")
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.show()

# Mechanical enregy
plt.plot(t, mechanical_energy(true_traj, params), label = "True")
plt.plot(t, mechanical_energy(mlp_traj, params),"--", label = "MLP")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.legend()
plt.show()

# Phase portrait
plt.plot(true_traj[:,0], true_traj[:,1], label = "True")
plt.plot(mlp_traj[:,0], mlp_traj[:,1],"--", label = "MLP")
plt.xlabel("Theta (rad)")
plt.ylabel("Omega (rad/s)")
plt.legend()
plt.show()
