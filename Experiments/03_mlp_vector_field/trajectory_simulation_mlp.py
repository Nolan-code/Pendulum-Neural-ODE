def trajectory_simulation_mlp(x0, dt, T):
    N = int(T / dt)
    x = torch.zeros((N+1, 2))
    x[0] = torch.tensor(x0, dtype=torch.float32)

    for t in range(N):
        x[t+1] = rk4_step_torch(f_hat, x[t], dt)

    return x.numpy()
