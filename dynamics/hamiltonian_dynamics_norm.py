def hamiltonian_dynamics_norm(model, x): # compute x' using the HNN, same use as hamiltonian_dynamics but x muste be shape (2,)
  x_n = (x - X_mean)/X_std
  x_n = x_n.clone().detach().requires_grad_(True)
  H_n = model(x_n).sum()

  grad_Hn = torch.autograd.grad(
      H_n,
      x_n,
      create_graph=True
  )[0]

  grad_H = grad_Hn / X_std
  d_theta = grad_H[1]
  d_omega = -grad_H[0]

  return torch.stack([d_theta, d_omega], dim=0)
