# Using a Deep Q Network

As a second experiment we tried to implement the agent using a *Deep Q Network (DQN)*.

We acheived this using the Torch framework. We also use a *Replay Memory* to improve the performance of the agent.

## The network

The objective of a Reinforcement Learning alorithm is that the Agent develops a deep knowledge of the environment so that it can take the best action in every situation.

In some environments, though, it is impossible to exhaustively explore the space of all the possible states and actions that can appear, as in the case of the cartpole game and of a number of more difficult games and real situations.

In these cases a neural network, that is a universal function approximator, can be used to estimate the Q* function, that is a function that, given the current state and an action returns the reward associated.

The deep network used by the agent to estimate the Q* function is composed by three convolutional layers with batch normalization and a linear layer with categorical output that represents the probability of each action.

The convolutional layers have size 16, 32, 32, while the linear layer have 448 units. We use the *Rectified Linear Unit* as activation function, the *Huber* loss function and the *RMSprop* optimizer.

The network is implemented in the `DQN` class. This inherit from the `Module` class of Torch.

## The Replay Memory

We will use an experience replay memory to train the Neural Network. 

The memory is implemented in the `ReplayMemory` class as a cyclic buffer that will keep the recently seen transitions between consecutive states. When the buffer size reaches its capacity, the oldest memories are overwritten. A `sample(batch_size)` method lets the user select a random batch of memories.

The `Transition` is a tuple containing the two states, the action that caused the transition itself and the reward associated to it.

We expect the random sampling to improve the training.

## The State
The state of the environment is an RGB image. In particular, we use a technique to give the Agent a perception of the speed.

First of all, the frame of the game is cropped so that the cart takes the biggest part of the image. Then, it is computed the difference with the previous frame.

This version of the state is a step toward an uninformed Agent, that would be able to play also different games. Still, it needs to know the cart location to be able to properly crop the image.

The state is retrieved by the agent using the `get_state()` method of itself, so that it is possible to change the type of information the DQN will process by just changing the content of this function.

## The Agent
The agent is implemented by the `Agent` class.

It contains methods to extract the state from the environment, to plot the duration of the episods it played, to select an action given a state and to train itself and optimize its DQN.

It can be configured using a `Config` object that specify learning rate, batch size, a seed (so that the behavior is deterministic between runs), and a preinizialized environment.

At the initialization it will create the DQNs and a replay memory. The Agent will use two DQNs, one is optimized after every step, while the other is updated less often.

### Action selection
The action is selected according to an ϵ-greedy policy. The agent will use the model to select the action, but sometimes it will take a random choise. The probability to chose at random will decrese exponentially.

### Training
The training is implemented in two methods: `train(num_epochs)` will run the training loop for the specified number of epochs. For each iteration, it will reset the environment and the state; than it will start playing. For every tik, the Agent will:
* take an action, 
* get the reward, 
* save the transition in the memory
* optimize the model.

The optimization of the model is performed in a separated function: `optimize_model()`. It will select a batch of memories excluding the ones that bring to the end of the match.

Then, it computes, using the current weights, the value of the Q function for the considered couples `(state, action)`. Using the next state, it is computed the value V for all of them; a value of 0 is assigned if next state is a game over.

The value V is computed using the `target_net`: this helps to improve stability as its weights are updated less often with respect to the policy_net.

At this point, the expected value of the Q funciton is computed and the loss.

Applying the backpropagation algorithm, the parameters of the policy net are updated and the optimization step ends.
