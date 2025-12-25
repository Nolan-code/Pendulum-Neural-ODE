### Third model
In this experiment, the neural network is trained to learn the Hamiltonian of the system. The state derivatives (d_theta, d_omega) are then recovered from the Hamiltonian using Hamilton’s equations, ensuring a conservative vector field by construction.

### Trajectory
- The trajectory build using the prediction of the HNN differs from the true trajectory despite the leanred Hamiltonian.

### Phase portrait
- The learned phase portrait clearly shows open trajectories whereas closed shape trajectories were exepted for a conservative system.

### Analysis
- Even thought the learned Hamiltonian is conserved, it does not correctly captures the physical strucure of the system.
- This discrepancy probably arises from the lack of physical constraint such as the periodicity of theta (and maybe the lack of data around the value of pi).
- As a result, the model learn a conserved quantity with the wrong typology leading to wrong trajectories.
