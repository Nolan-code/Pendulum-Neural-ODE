### Third model
In this experiment, the neural network is trained to learn the Hamiltonian of the system. The state derivatives are then recovered using Hamilton’s equations. Compared to the previous experiments, the training dataset was modified.
The dataset was constructed such that approximately 1/50 of the samples correspond to low-energy trajectories (about 300 points). The train–test split was designed to ensure that low, medium, and high energy trajectories are present in both the training and test sets. Moreover, a new loss term was introduced in an attempt to force the neural network to place the stable equilibrium point at θ=π.

#### Trajectory
- Despite failing to learn that the stable equilibrium point is located at pi, the neural network is able to generate physically consistent trajectories. In particular, it successfully reproduces the correct oscillations amplitude and period, even for initial condition away from the equilibrium point.

![Trajectory](figures/HNN_v1.png)


#### Phase portrait
- The neural network is now able to generate closed phase space trajectories when the system does not have sufficent energy to do a full rotation, which is consistent with the expected behavior of the pendulum.

![Trajectory](figures/HNN_phase_v1.png)

#### Analysis
- Despite explicitly penalizing incorrect equilibrium positioning through a new loss term, the neural network is consistently positioning the equilibrium point at θ=0 instead of θ=π.
- This behavior clearly shows an identifiable issue: while the learned Hamiltonian produces correct dynamics, the position of the equilibrium point is not uniquely determined by the equations of motion alone.

#### Conclusion
- This final HNN experiment demonstrates that:
  - Hamiltonian Neural Networks can successfully learn locally accurate conservative dynamics
  - However, without additional global constraints, they may fail to recover:
    - The correct energy landscape
    - The true location of stable equilibrium point
- These limitations motivate the transition to another model, a Lagrangian neural network where:
  - The potential energy is directly learned
  - Stable equilibria correspond to minima of a learned scalar function
  - The physical structure of the system is more strongly constrained
This clearly exhibits a limitation of the HNN, the equation of motion constrain gradient of the Hamiltonian but not it's absolute shape or it's stable equilibrium point location.
