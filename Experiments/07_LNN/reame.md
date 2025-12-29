# Lagrangian Neural Network (LNN)
While Hamiltonian neural network succesfully conserved the energy of the system, it still seems to lack physical constraints to place the stable equilibrium point of the pendulum accurately, that's why I am now switching to a Lagrangian neural network to add more physical contraints.

## Lagrangian formulation
- The dynamics are derived from the Lagrangian:
    L(θ, ω) = T(ω) - V(θ)
- The equations of motion follow from the Euler–Lagrange equations:
    d/dt (∂L/∂ω) - ∂L/∂θ = 0
