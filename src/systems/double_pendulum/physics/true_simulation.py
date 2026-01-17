import torch
import numpy as np

def double_pendulum_dynamics(X, params):  # Compute the function f(x,t) such that theta'' = f(x,t)
  if X.dim() == 1:
      X = X.unsqueeze(0)
  theta1, theta2, omega1, omega2 = X[:,0], X[:,1], X[:,2], X[:,3]
  m, l, g = params["m"], params["l"], params["g"]

  omega1_dot = (-g * (2 * torch.sin(theta1) - torch.sin(theta2) * torch.cos(theta1 - theta2)) - torch.sin(theta1 - theta2) * (omega2**2 * l + omega1**2 * l * torch.cos(theta1 - theta2))) / (l * (2 - torch.cos(theta1 - theta2)**2))
  omega2_dot = (2 * torch.sin(theta1 - theta2) * (omega1**2 * l * (1) + g * torch.cos(theta1)) + omega2**2 * l * torch.sin(theta1 - theta2) * torch.cos(theta1 - theta2)) / (l * (2 - torch.cos(theta1 - theta2)**2))

  dX = torch.stack([omega1, omega2, omega1_dot, omega2_dot], dim=1)
  if X.dim() == 1:
      dX = dX.squeeze(0)
  return dX

def rk4_step(f, x, dt, params):  # Compute the position after a short time dt based on the pervious position using rk4 method
  dt_tensor = torch.tensor(dt, dtype=torch.float64)

  k1 = f(x, params)
  k2 = f(x + 0.5 * dt_tensor * k1, params)
  k3 = f(x + 0.5 * dt_tensor * k2, params)
  k4 = f(x + dt_tensor * k3, params)

  return x + (dt_tensor / 6) * (k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation(x0, dt, T, params):
  N = int(T/dt)
  x = torch.zeros((N+1, 4), dtype=torch.float64)
  x[0] = torch.tensor(x0, dtype=torch.float64)

  for t in range(N):
   x[t+1] = rk4_step(double_pendulum_dynamics, x[t], dt, params)
  return x.numpy() # Convert the final trajectory back to a numpy array
