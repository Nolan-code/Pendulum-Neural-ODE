def pendulum_dynamics(x, u, params):  # Compute the function f(x,t) such that theta'' = f(x,t)
  theta, omega = x
  m, l, g = params["m"], params["l"], params["g"]
  u = float(u)

  d_theta = omega
  d_omega = (g/l) * np.sin(theta) + u/(m * l**2)

  return np.array([d_theta, d_omega])

def rk4_step(f, x, u, dt, params):  # Compute the position after a short time dt based on the pervious position using rk4 method
  k1 = f(x, u, params)
  k2 = f(x + 0.5 * dt * k1, u, params)
  k3 = f(x + 0.5 * dt * k2, u, params)
  k4 = f(x + dt * k3, u, params)

  return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation(x0, u_control, dt, T, params):  # Compute the trajectory using rk4_step at each step
  N = int(T/dt)
  x = np.zeros((N+1, 2))
  x[0] = x0

  for t in range(N):
    u = u_control(t * dt, x[t])
    x[t+1] = rk4_step(pendulum_dynamics, x[t], u, dt, params)
  return x

def zero_control(t,x):
  return 0


def simulate(model, x0, dt, T, format):
    n_steps = int(T / dt)
    traj = []

    x = torch.tensor(x0, dtype=torch.float32) 
    if x.dim() == 1:
        x = x.unsqueeze(0) 

    for _ in range(n_steps + 1):
        traj.append(x.clone().detach()) 

        k1 = model(model_input(x, format)).detach() 
        k2 = model(model_input(x + dt/2 * k1, format)).detach()
        k3 = model(model_input(x + dt/2 * k2, format)).detach()
        k4 = model(model_input(x + dt * k3, format)).detach()

        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4) 

    return torch.cat(traj, dim=0).cpu().numpy()

def model_input(x,format):
  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x, dtype=torch.float32)

  if x.dim() == 1:
    x = x.unsqueeze(0) 

  if format == "sincos":
      sin = torch.sin(x[:,0]).unsqueeze(1)
      cos = torch.cos(x[:,0]).unsqueeze(1)
      omega = x[:,1].unsqueeze(1)
      x = torch.cat([sin, cos, omega], dim=1)
  return x
