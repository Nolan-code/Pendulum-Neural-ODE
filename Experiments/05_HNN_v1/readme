### Forth model
In this experiment, the neural network is trained to learn the Hamiltonian of the system. The state derivatives are then recovered using Hamiltonian's equations. However, in this experiment the dataset used for training was modified. 
The dataset was built in a way such that 1/50 of the point are from low energy trajectory (it represent roughly 300 points) and the split between train and test was made such that there are low, mid and hight energies trajectory in both 
the training set and test set. Moreover a new loss was introduced to try to force the neural network to place the stable equilibrium point at pi.

#### Trajectory
- The neural network, despite despite failing to learn that the stable equilibrium point is at pi, managed to build physicly correct trajectories. In fact, it is able to get the correct amplitude and period of the oscillations.

#### Phase portrait
- The neural network is now able to build close shaped phase portrait when the system does not have the required amount of energy to do a full revolution.

#### Analysis
- Despite trying to force the system to learn the position of the stable equilibrium point by adding a loss that penalize a wrong position of it, the neural network fails to learn it's correct position and is placing it at 0.
