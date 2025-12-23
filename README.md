# Inverted-Pendulum-Neural-ODE
This project investigates the learning of nonlinear physical dynamics using neural networks, with the inverted pendulum as a benchmark system.

## Physics validation of the rk4 method
We first validate the numerical integration of the inverted pendulum dynamics by analyzing energy conservation and phase-space structure. The goal of this step is to ensures that the simulator provides physically consistent trajectories before any learning stage.
- ![Mecganical energy conservation](figures/E_mech_conservation.png)

## Results of the MLP
### One-step prediction
- The MLP achieves low error in the one-step prediction task.
- Both training and test losses converge quiclky toward 1e-3, 1e-4.

### Long-term rollout
- The learned dynamics reproduces the qualitative behavior of the trajectory.
- However, the amplitude and frequency of the trajectory is diverging from the true dynamics over time.
![Mecganical energy conservation](figures/MLP_vs_true.png)

### Energy analysis
- The true system conserves mechanical energy.
- The MLP mechanical energy tend to diverge over time.

### Phase portrait
- The phase-space structure is approximately captured.
- The long-term trajectories diverge from the expected dynamics due to error accumulation.
- The lack of closed trajectories in the learned phase portrait is a consequence of the model not conserving the mechanical energy.
![Mecganical energy conservation](figures/phase.png)

## Rollout with RK4
- In this experiment, the MLP learn how to predict the state derivatives (d_theta, d_omega) from the state (theta, omega).The vector field is then integrated using RK4 method to generate the trajectory.

### Trajectory
- The predicted trajectory using MLP differs significantly from the true trajectory. In particular, the learned dynamics shows a stable equilibrium point at θ=0, whereas the true pendulum has its stable equilibrium at θ=π. Moreover, the amplitude of the oscillations decreases over time despite the data being generated from a conservative system.
- ![Trajectories](figures/mlp_rk4.png)

### Phase portrait
- The learned phase portrait clearly shows a spiral-like pattern converging toward a single attractor at (0,0), which indicating artificial energy dissipation. This behavior arises from the lack of physical constraints such as energy conservation in the MLP architecture. As a result, despite achieving a low local prediction error, the learned dynamics fail to reproduce the correct long-term qualitative behavior of the system.
![Vector field of the MLP](figures/vector_mlp.png)

  
