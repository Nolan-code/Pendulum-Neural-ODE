def trajectory_simulation_HNN(x0, dt, T):  # adapted version for the HNN
    N = int(T / dt)
    x = torch.zeros((N+1, 2))
    x[0] = torch.tensor(x0, dtype=torch.float32)
    E = np.zeros(N+1)
    E[0] = (1/2)*(x0[1]**2) + (1 - np.cos(x0[0]))

    for t in range(N):
        x[t+1] = rk4_step_HNN(model, x[t], dt).detach()
        E[t+1] = (1/2)*(x[t+1,1]**2) + (1 - np.cos(x[t+1,0]))

    return x.detach().numpy(), E
