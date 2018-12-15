# Using a Deep Q Network

As a second experiment we tried to implement the agent using a *Deep Q Network (DQN)*.

We acheived this using the Torch framework. We also use a *Replay Memory* to improve the performance of the agent.

## The network

The deep network used by the agent to estimate the Q function is composed by three convolutional layers with batch normalization and a linear layer with categorical output that represents the probability of each action.

The convolutional layers have size 16, 32, 32, while the linear layer have 448 units. We use the *Rectified Linear Unit* as activation function and the RMSprop optimizer.

The network is implemented in the `DQN` class. This inherit from the `Module` class of Torch.

## The State
The state of the environment is an RGB image. In particular, we use a technique to give the Agent a perception of the speed.

First of all, the frame of the game is cropped so that the cart takes the biggest part of the image. Then, it is computed the difference with the previous frame.

This version of the state is a step toward an uninformed Agent, that would be able to play also different games. Still, it needs to know the cart location to be able to properly crop the image.

The state is retrieved by the agent using the `get_state()` method, so that it is possible to change the type of information the DQN will process by just changing the content of this function.

## The Agent
