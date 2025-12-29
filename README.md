# Inverted-Pendulum-Neural-ODE
This project investigates the learning of nonlinear physical dynamics using neural networks, with the inverted pendulum as a benchmark system. The goal is to compare different learning strategies—from unconstrained neural networks to physics-informed architectures and analyze their ability to reproduce the qualitative and quantitative properties of a conservative dynamical system.

## Physical system
We considered a planar pendulum governed by a non-linear dynamics. The system is conservative and admits a Hamiltonian formulation, making it ideal for physical informed learning. The state is given by x = (θ, ω) with θ the angle and ω the angular velocity.

## Numerical baseline
Before any learning stage, the dynamics are generated using the RK4 step approximation method (4th order). The energy conservation and the phase-space structure are verified to ensure physically consistent trajectories.

## Learning approach
I progressively increased the amount of physical constraints imposed by the model:
- Direct state prediction using an MLP
- learning the vector field and integrating using RK4
- Hamiltonian neural network (HNN)
- Lagrangian neural network (LNN)

## Key elements
- Purely data-driven (without physical constraints) model achieves low quality predictions, especially one-step prediction which rapidly differs from the true trajectory due to error accumulation.
- Learning the vector field improves short-term prediction but also introduces artificial energy dissipation, causing the amplitude of the oscillations to decrease over time.
- Physics informed models enforce global strucure and also help to diagnose inconsistencies in the training data.
- Correct physically modeling requires both well-structured models with physical constraints and physically consistent data.

## Lessons learned
This project highlights that physics informed models are not only a predictive tool but also a good way of diagnosing physical inconstitencies in the training data. In particular, the Hamiltonian model exhibits signs of physical data inconstitencies that weren't detected by standard neural networks.


