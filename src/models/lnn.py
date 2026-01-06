class LNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.V = MLP_V()

  def lagrangian(self, X):
    if X.dim() == 1:
        X = X.unsqueeze(0)
    sin, cos, omega = X[:,0].unsqueeze(1), X[:,1].unsqueeze(1), X[:,2].unsqueeze(1)
    T = (1/2) * omega**2
    V = self.V(torch.cat([sin, cos], dim=1))
    L =  T - V
    return L

  def forward(self, X):
    if X.dim() == 1:
        X = X.unsqueeze(0)
    X = X.requires_grad_(True)
    sin, cos, omega = X[:,0].unsqueeze(1), X[:,1].unsqueeze(1), X[:,2].unsqueeze(1)
    L =  self.lagrangian(X)

    grad = torch.autograd.grad(L.sum(), X, create_graph=True)[0]

    dL_dsin = grad[:,0]
    dL_dcos = grad[:,1]
    dL_dw   = grad[:,2]

    dL_dtheta = dL_dsin * X[:,1] - dL_dcos * X[:,0]
    dL_domega = dL_dw

    d2L_d2omega = torch.autograd.grad(dL_domega.sum(), X, create_graph=True)[0][:,2].squeeze()

    eps = 1e-6
    domega = dL_dtheta / (d2L_d2omega + eps)
    dtheta = omega.squeeze()

    dx = torch.column_stack([dtheta, domega])
    return dx

class MLP_V(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
    )

  def forward(self, x):
    return self.net(x)
