# Experimenting with Reinforce algorithm: Deep Policy Gradient Agent

After adapting the code from Victor Campos (DLAI lab) to PyTorch we started modifying some hyperparameters/architecture.
Here you can find a small resume of the most representative experiments:

## Observations as current state 

1. Activation function: ReLU vs TanH
relu_vs_tanh.png
Basic Deep Policy Gradient Agent with lr=0.001 using a 3 layer FC network.

As can be seen in the following image, the policy which uses ReLU activations needs more episodes to perform as good as the one with tanh activations. However, ReLU agent shows a better convergence at the end of the training.
It can be explained by the information ReLU is dropping (0ing everything below 0).

![Octocat](assets/images/relu_vs_tanh.png)



2. Entropy normalization
Exploration is one of the problems one has to face when working on reinforcement learning. Intuitively, encouraging our agent to explore the environment can improve the speed of the training and avoid falling in non-optimal local minima.
More information can be found in the following paper:
Understanding the impact of entropy on policy optimization, Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, Dale Schuurmans
(Submitted on 27 Nov 2018 (v1), last revised 29 Nov 2018 (this version, v2)) 

After implementing entropy normalization with a constant value we can see the following results:

## RGB Image as current state