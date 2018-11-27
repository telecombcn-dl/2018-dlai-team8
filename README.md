# 2018-dlai-team8: 
## Policy-based reinforment algorithms
Interesting links:
https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f 

### Experiment 1: Notebook Exp1 --> cartpole env
The first experiment we will carry out will be adapting the code from lab06 reinforment learning by Victor Campos to work with PyTorch.

- Critical part: Adapting the policy gradient update!!!!

Some agent's rewards:
- 1500 episodes with batch_size=1 and lr=0.001
Blue -> No entropy

<img src="captures/1500ep_1bs_0.001lr.png"/>

- 3500 episodes with batch_size=2 and lr=0.001

<img src="captures/3500ep_2bs_0.001.png"/>

### Experiment 2: Notebook Exp2 --> cartpole env
The second experiment will consist on adding a replay memory to the agent, and sample data from it to train.

### Experiment 3: Notebook Exp3 --> lunar lander env
The third experiment will consist on adapting the agent developed before in a more challenging environment
Bigger observation space + Bigger action space (also discrete)


### Experiment 4: Notebook Exp4 --> cartpole env
The fourth experiment will consist on adapting the agent developed before in Exp1 to work directly with the environment images instead of the observation space.

