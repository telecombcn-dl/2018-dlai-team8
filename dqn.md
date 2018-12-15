# Using a Deep Q Network

As a second experiment we tried to implement the agent using a *Deep Q Network (DQN)*.

We acheived this using the Torch framework. We also use a *Replay Memory* to improve the performance of the agent.

## The network

The deep network used by the agent to estimate the Q function is composed by three convolutional layers with batch normalization and a linear layer with categorical output that represents the probability of each action.

The convolutional layers have size 16, 32, 32, while the linear layer have 448 units. We use the *Rectified Linear Unit* as activation function.

## The State
The state of the environment is an image. In particular, we use a 
