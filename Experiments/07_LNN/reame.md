### Lagrangian Neural Network (LNN)
While Hamiltonian neural networks successfully conserved the energy of the system, it still seems to lack physical constraints to place the stable equilibrium point of the pendulum accurately. That's why I am now switching to a Lagrangian neural network to add more physical constraints. The goal of the LNN is to compute L = T(ω) - V(θ) and then recover the state derivatives using Lagrange equations.

#### Lagrangian formulation
- The dynamics are derived from the Lagrangian:
    L(θ, ω) = T(ω) - V(θ)
- The equations of motion follow from the Euler–Lagrange equations:
    d/dt (∂L/∂ω) - ∂L/∂θ = 0
#### Dynamics
- The LNN reproduces the dynamics of the system consistently after correction of the dataset. It is now able to place accurately the stable equilibrium point at π. The LNN is also able to generate physically consistent trajectories even over long periods of time (more than 60 seconds), unlike the one-step prediction model, which was only able to reproduce the shape of the trajectory but not the amplitude due to energy loss over time.
![Trajectory](figures/trajectory_LNN.png)
![Phase portrait](figures/phase_portrait_LNN.png)


