import torch
import torch.nn as nn

class LNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.V = MLP_V()

    def lagrangian(self, q, qd):
        theta1, theta2 = q[:, 0], q[:, 1]
        omega1, omega2 = qd[:, 0], qd[:, 1]

        delta = theta1 - theta2

        T = (
            omega1**2
            + 0.5 * omega2**2
            + omega1 * omega2 * torch.cos(delta)
        )

        V = self.V(torch.stack([
            torch.sin(theta1), torch.cos(theta1),
            torch.sin(theta2), torch.cos(theta2)
        ], dim=1)).squeeze()

        return T - V

    def forward(self, X):
        q = X[:, :2].clone().requires_grad_(True)
        qd = X[:, 2:].clone().requires_grad_(True)

        L = self.lagrangian(q, qd)

        # ∂L/∂q
        dL_dq = torch.autograd.grad(
            L.sum(), q, create_graph=True
        )[0]

        # ∂L/∂qd
        dL_dqd = torch.autograd.grad(
            L.sum(), qd, create_graph=True
        )[0]

        # Mass matrix M = ∂²L / ∂qd²
        M = torch.zeros(X.shape[0], 2, 2, device=X.device)
        for i in range(2):
            grad = torch.autograd.grad(
                dL_dqd[:, i].sum(), qd, create_graph=True
            )[0]
            M[:, i, :] = grad

        # Coriolis / potential term: ∂²L / ∂qd∂q · qd
        C = torch.zeros(X.shape[0], 2, device=X.device)
        for i in range(2):
            grad = torch.autograd.grad(
                dL_dqd[:, i].sum(), q, create_graph=True
            )[0]
            C[:, i] = (grad * qd).sum(dim=1)

        qdd = torch.linalg.solve(M, (dL_dq - C).unsqueeze(-1)).squeeze(-1)

        return torch.cat([qd, qdd], dim=1)
class MLP_V(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
