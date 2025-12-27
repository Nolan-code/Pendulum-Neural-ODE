### Fifth model
In this experiment, the architecture of the model has changed. In the previous model, it compute directly the hamiltonian based on (sin, cos, w) while this time, the model will separately compute the two component of the Hamiltonian. The first will try to learn (1/2) * w**2 only knowing w and the second one will learn 1 - cos(theta) only knowing sin(theta) and cos(theta).

#### Analysis
- The learned Hamiltonian is defined up to a symmetry in the angular coordinate. While the model reproduces the correct dynamics, the absolute position of the stable equilibrium is not identifiable from data alone.

#### Conclusion
- The Hamiltonian learned is defined to within one symmetry; the origin of the potential cannot be identified from the trajectories.
