---
layout: default
---


# Introduction to Reinforcement Learning

bla bla bla intro

# Deep Stochastic Policy Gradient Agent

## Deterministic vs Stochastic

Depending on how the policy is defined classify agents as:
* Deterministic: Outputs the action the agent will execute for a given state. Value-based agents such as DQN are exmples of deterministic agents.

* Stochastic: Outputs a probability distribution over actions. The action executed is sampled from this distribution

## Policy definition:
We wont our policy to learn the probability distribution for a given state, this will be achieved using a fully connected neural network

What to learn?
![Octocat](assets/images/stochastic.png)

Who will learn it? Our model


```python
# Pytorch policy definition
class PolicyNN(nn.Module):
  def __init__(self,obs_dim=4, num_actions=2):
    super(PolicyNN, self).__init__()
    self.fc1 = nn.Sequential(
      nn.Linear(obs_dim, 16),
      nn.Tanh())
    self.fc2 = nn.Sequential(
      nn.Linear(16, 16),
      nn.Tanh())
    self.fc3 =  nn.Linear(16, num_actions)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    logprobs = self.fc3(x)
    return F.softmax(logprobs)
}
```


That's perfect, but how do we know how good is our policy? 

As it is an optimization problem we need to find a mesure to minimize/maximize and update our policy accordingly.


## Policy Optimization:
Basic definitions:


![Octocat](assets/images/update_alg.png)

How to train the policy:

1. Measure the quality of a π (policy) that has parameters θ with a policy score function J(θ) 
2. Use policy gradient ascent to find the best parameter θ that improves our π
    

Defining J(θ):

The main idea behind reinforcement learning is the idea of the reward hypothesis. It says that all goals can be descibed by the maximization of the expected cumulative reward:

- We can define the policy score as the expected reward of following π for every possible state:

![Octocat](assets/images/score_function.png)

We know that policy parameters change how actions are chosen, and as a consequence, what rewards we get and which states we will see and how often.
On the other hand, the impact of the policy in the state distribution is not that obvious, moreover the environment is unknown


The solution will be to use the Policy Gradient Theorem. This provides an analytic expression for the gradient ∇ of J(θ) (performance) with respect to policy θ that does not involve the differentiation of the state distribution.

Differentiable score policy function:

![Octocat](assets/images/differentiable_score.png)

Now we can compute the compute the gradient and define the updates as:

![Octocat](assets/images/grad_score.png)

So, what we need to train our policy:
- Trajectories: pairs state, action, next_state
- Rewards: discounted rewards

This expression can bem easily implement in pytho as:

```python
# Pytorch Score function
    for i in reversed(range(len(rewards))):
        # Discounted reward
        R = self.gamma * R + rewards[i]
        # log_probs = log (P(action|state)) 
        loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum()
          
}
```
Note that this maximization problem is swapped to minimization by changing the reward sign in order to use autogran and torch optimizers.

Finally we can update the parameters π(θ) following Monte Carlo update:

![Octocat](assets/images/update.png)


## 



# DQN Agent

balblabalba


# Conclusions

balblabalba

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
