def hamiltonian_dynamics(model, X_tilde, theta):
    X_tilde = X_tilde.requires_grad_(True)
    H = model(X_tilde).sum()

    grad = torch.autograd.grad(H, X_tilde, create_graph=True)[0]

    dH_dsin = grad[:,0]
    dH_dcos = grad[:,1]
    dH_dw   = grad[:,2]

    dH_dtheta = dH_dsin * torch.cos(theta) - dH_dcos * torch.sin(theta)

    dtheta = dH_dw
    domega = -dH_dtheta

    return dtheta, domega

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

def hamiltonian_dynamics_simulation(model, x):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    theta = x[:,0]
    omega = x[:,1]

    X_tilde = torch.stack([torch.sin(theta), torch.cos(theta), omega], dim=1)
    X_tilde = X_tilde.requires_grad_(True)
    H = model(X_tilde).sum()

    grad = torch.autograd.grad(H, X_tilde, create_graph=True)[0]

    dH_dsin = grad[:,0]
    dH_dcos = grad[:,1]
    dH_dw   = grad[:,2]

    dH_dtheta = dH_dsin * torch.cos(theta) - dH_dcos * torch.sin(theta)

    dtheta = dH_dw
    domega = -dH_dtheta

    dx = torch.stack([dtheta, domega], dim=1)

    return dx.squeeze(0) if dx.shape[0] == 1 else dx


def rk4_step_HNN(f, x, dt):   # rk4_step but adapted for the HNN
    k1 = hamiltonian_dynamics_simulation(f, x)
    k2 = hamiltonian_dynamics_simulation(f, x + 0.5 * dt * k1)
    k3 = hamiltonian_dynamics_simulation(f, x + 0.5 * dt * k2)
    k4 = hamiltonian_dynamics_simulation(f, x + dt * k3)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation_HNN(x0, dt, T):  # adapted version for the HNN
    N = int(T / dt)
    x = torch.zeros((N+1, 2))
    x[0] = torch.tensor(x0, dtype=torch.float32)

    for t in range(N):
        x[t+1] = rk4_step_HNN(model, x[t], dt).detach()

    return x.detach().numpy()

def dH_dt(model, x):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    theta = x[:, 0]
    omega = x[:, 1]

    X_tilde = torch.stack(
        [torch.sin(theta), torch.cos(theta), omega],
        dim=1
    ).requires_grad_(True)   # network input

    H = model(X_tilde).sum() # hamiltonian

    grad = torch.autograd.grad(H, X_tilde, create_graph=True)[0]

    dH_dsin = grad[:, 0]
    dH_dcos = grad[:, 1]
    dH_dω   = grad[:, 2]

    dH_dθ = dH_dsin * torch.cos(theta) - dH_dcos * torch.sin(theta) # chain rule

    dθ = dH_dω
    dω = -dH_dθ

    dHdt = dH_dθ * dθ + dH_dω * dω

    return dHdt.squeeze()


def mechanical_energy(x, params):    # Verification that Em is roughly constant
  m, l, g = params["m"], params["l"], params["g"]
  theta = x[:,0]
  omega = x[:,1]

  E = 0.5 * m * l**2 * omega**2 + m * g * l * np.cos(theta)
  return E
