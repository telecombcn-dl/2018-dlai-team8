# Conclusions

In this project, the team had the first approach to a complete Reinforce Learning project. Being able to study different algorithms, test different architectures and trying to get the most out of each algorithm.

Focusing mainly on Deep Policy Gradient Agent at the start point, it was possible to train a network that is able to obtain the maximum reward taking as input the discrete state of the Cart and pole environment. After this first satisfactory result, we could demonstrate the importance of the exploration and the selection of an activation function. Comparing the results using entropy normalization which increases the speed and accuracy when converging. And the results using ReLu or TanH where the first one increases the final convergence of the model whereas the second one has a higher speed at the beginning.

In the second part, we faced the problem of having a more realistic input to the agent that was using an RGB image as the state of the environment. In this case, the results were not so good as the results with the discrete state. The problem complexity increased markedly. Adapting the network resulting from the previous experiment, to a CNN plus a Fully Connected one was not enough and the model did not converge. And we neither be able to converge trying a DQN.

Although we could not reach all the goals of the project, we are happy with the results because we knew it was a challenging project and even so we had the opportunity to learn a lot about deep learning and more specifically about Reinforcement learning.

# Future Work

Due to lack of time, there were some alternatives and experiments we could not perform and we like to mention them in order to explain the possible solutions we are going to try whenever we have time.
- **Redesign of the network**: increase number of FC layers.
- **Try different policy optimization algorithms**: like Trust-Region Policy Optimization
- **Perform the train in a different environment**: once we overcome the Cart and pole environment using as state the RGB images, try to solve another environment like the lunar lander.

