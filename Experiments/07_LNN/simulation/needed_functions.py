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

def lagrangian_dynamics(model, theta, omega):
  theta = theta.requires_grad_(True)
  omega = omega.requires_grad_(True)

  L = model(theta,omega).sum()

  dL_dtheta = torch.autograd.grad(L, theta, create_graph=True)[0]
  dL_domega = torch.autograd.grad(L, omega, create_graph=True)[0]

  d2L_d2omega = torch.autograd.grad(dL_domega.sum(), omega, create_graph=True)[0]
  
  eps = 1e-6
  domega = dL_dtheta / (d2L_d2omega + eps)
  dtheta = omega

  return dtheta,domega

def rk4_step_LNN(model, x, dt):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    def f(x):
        theta, omega = x[:,0], x[:,1]
        dtheta, domega = lagrangian_dynamics(model, theta, omega)
        return torch.stack([dtheta, domega], dim=1)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation_LNN(x0, dt, T):  # adapted version for the HNN
    N = int(T / dt)
    x = torch.zeros((N+1, 2))
    x[0] = torch.tensor(x0, dtype=torch.float32)

    for t in range(N):
        x[t+1] = rk4_step_LNN(model, x[t], dt).detach()

    return x.detach().numpy()
