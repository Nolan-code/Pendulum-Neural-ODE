def rk4_step_HNN(f, x, dt):   # rk4_step but adapted for the HNN
    k1 = hamiltonian_dynamics_norm(f, x)
    k2 = hamiltonian_dynamics_norm(f, x + 0.5 * dt * k1)
    k3 = hamiltonian_dynamics_norm(f, x + 0.5 * dt * k2)
    k4 = hamiltonian_dynamics_norm(f, x + dt * k3)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
