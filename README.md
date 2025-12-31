# Inverted-Pendulum-Neural-ODE
This project investigates the learning of nonlinear physical dynamics using neural networks, with the inverted pendulum as a benchmark system.   
The goal is to compare purely data-driven models to physics informed architectures and to analyze their ability to reproduce both the quantitative accuracy and qualitative physical structure of a conservative dynamical system.

## Physical system
We considered a planar pendulum governed by a non-linear second order ordinary differential equation. The system is conservative and admits a Hamiltonian formulation, making it suitable for physical informed learning.   
The state is given by:    
- x = (θ, ω)     
where :  
- θ the angle
- ω the angular velocity

The dynamics can be written as:   
- θ' = ω
- ω' = (g/l) * ​sin(θ)

The Hamiltonian of the system is given by:  
- H(θ,ω) = (1/2) * ​ω2 + (g/l) * (1 − cosθ)

## Numerical baseline
Before any learning stage, references trajectories are generated using a forth order Runge–Kutta (RK4) integration scheme.   
The numerical solver is validated by checking :
- the energy conservation
- phase-space structure
- long term qualitative behavior    
This ensure that the training data is physically consistent.

## Learning approach
Several learning strategies with increasing level of physical structure are compared.
#### Direct state prediction using an MLP
A multilayer perceptrons predicts the next state x(t+1) from thecurrent state x(t). This approach is purely data driven and do not impose any physical constraints.
#### Learning the vector field and integrating using RK4
The neural network learns the vector fields x'=f_θ(x). The prediction is then recovered by integrating this vector field by using RK4.
#### Hamiltonian neural network (HNN)
The neural network learns a scalar Hamiltonian function H_θ(x), from which the dynamics are derived using:
- x'=J∇H_θ​(x)   
where J is the canonical symplectic matrix.
This structure aim to force energy conservation of the system.
#### Lagrangian neural network (LNN)
The dynamics are learn via the Lagrangian L_θ​(q,q'​), using the Euler-Lagrange equations.

## Key elements
- Purely data-driven model fails to conserve the qualitative structure of the dynamics and quickly differs from the true behavior due to error accumulation
- Learning the vector field improves short-term prediction but also introduces artificial energy dissipation, causing the amplitude of the oscillations to decrease over time.
- Physics informed models enforce global structural constraints and achieve a better long term prediction
- These models are also extremly sensitive to inconsistencies in the training dataset

## Impact of dataset correction
During the training of the physics-informed models, inconsistencies were identified in the original dataset, notably in the sign and qualitative behavior of the angular acceleration.   
After correcting the dataset, earlier models (MLP-based predictors and HNN) were retrained for comparison.
- MLP based models improved quantitatively but still fail to preserve the correct physical structure
- Physics informed models, in comparaison, significantly improved and clearly predict the correct dynamics
This highlights an important distinction:   
purely data-driven models may fit inconsistent data without exposing underlying physical errors, whereas physics-informed architectures can clearly exhibit physical inconsistencies in the training dataset.

## Lessons learned
This project highlights that physics informed models are not only a predictive tool but also powerfull tools for:
- Diagnosing physical inconstitencies in the training data.
- Enforcing physical consistency
- preserving the dynamical structure over a long period of time

## Perspectives
Future extensions include:
- chaotic system
- dissipative system
- comparaison with PINNs 

