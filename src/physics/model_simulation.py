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
