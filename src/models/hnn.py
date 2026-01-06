import torch
import torch.nn as nn

class MLP_T(nn.Module):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(1,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,1)
    )

  def forward(self,x):
    return self.net(x)

class MLP_V(nn.Module):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(2,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,1)
    )

  def forward(self,x):
    return self.net(x)

class HNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = MLP_T() 
        self.V = MLP_V()  


    def hamiltonian(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)        
        sinθ, cosθ, omega = X[:,0].unsqueeze(1), X[:,1].unsqueeze(1), X[:,2].unsqueeze(1)
        T = self.T(omega)
        V = self.V(torch.cat([sinθ, cosθ], dim=1))
        return T + V

    def forward(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.requires_grad_(True)
        H = self.hamiltonian(X)

        grad = torch.autograd.grad(H.sum(), X, create_graph=True)[0]

        dH_dsin = grad[:,0]
        dH_dcos = grad[:,1]
        dH_dw   = grad[:,2]

        dH_dtheta = dH_dsin * X[:,1] - dH_dcos * X[:,0]

        grad = torch.stack([dH_dtheta, dH_dw], dim=1)

        # canonical symplectic matrix
        J = torch.tensor([[0., 1.],
                          [-1., 0.]],
                          device=X.device)

        dx = grad @ J.T
        return dx
