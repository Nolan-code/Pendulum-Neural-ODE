# Inverted-Pendulum-Neural-ODE
This project investigates the learning of nonlinear physical dynamics using neural networks, with the inverted pendulum as a benchmark system.

## Physics validation of the rk4 method
We first validate the numerical integration of the inverted pendulum dynamics by analyzing energy conservation and phase-space structure. The goal of this step is to ensures that the simulator provides physically consistent trajectories before any learning stage.
![Mecganical energy conservation](figures/E_mech_conservation.png)

## Results of the MLP
### One-step prediction
- The MLP achieves low error in the one-step prediction task.
- Both training and test losses converge quiclky toward 1e-3, 1e-4.

### Long-term rollout
- The learned dynamics reproduces the qualitative behavior of the trajectory.
- However, the amplitude and frequency of the trajectory is diverging over time.

### Energy analysis
- The true system conserves mechanical energy.
- The MLP mechanical energy tend to diverge over time.

### Phase portrait
- The phase-space structure is approximately captured.
- The long-term trajectories diverge due to error accumulation.
