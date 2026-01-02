import numpy as np
import torch

def model_input_hnn(x):
  sin = torch.sin(x[:,0].unsqueeze(1))
  cos = torch.cos(x[:,0].unsqueeze(1))
  omega = x[:,1].unsqueeze(1)
  x = torch.cat([sin, cos, omega], dim=1)
  return x
  
def rk4_step_HNN(model, x, dt):   # rk4_step but adapted for the HNN
    k1 = model( model_input_hnn(x) )
    k2 = model( model_input_hnn(x + 0.5 * dt * k1) )
    k3 = model( model_input_hnn(x + 0.5 * dt * k2) )
    k4 = model( model_input_hnn(x + dt * k3) )

    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation_HNN(x0, dt, T):  # adapted version for the HNN
    N = int(T / dt)
    x = torch.zeros((N+1, 2))
    x[0] = torch.tensor(x0, dtype=torch.float32)

    for t in range(N):
        x_current = x[t].unsqueeze(0)
        x[t+1] = rk4_step_HNN(model, x_current, dt).detach()

    return x.detach().numpy()
