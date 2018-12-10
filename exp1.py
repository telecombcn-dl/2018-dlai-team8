#!/usr/bin/python3
import os
import torch  # Check if everything is OK
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as utils
import sys

import gym
import numpy as np

# Some stuff that we will need to visualize the simulations in the notebook
import matplotlib.pyplot as plt

# Start virtual display
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)


class PolicyNN(nn.Module):
    def __init__(self, obs_dim=4, num_actions=2):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.Tanh())
        self.fc2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh())
        self.fc3 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        logprobs = self.fc3(x)
        return F.softmax(logprobs)


class PolicyNN_V2(nn.Module):
    def __init__(self, obs_dim=4, num_actions=2):
        super(PolicyNN_V2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU())
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        logprobs = self.fc3(x)
        return F.softmax(logprobs)


class Agent:
    def __init__(self, config, use_entropies=True, decay=False):
        self.env = gym.make(config.env)
        self.pi = PolicyNN_V2()
        self.gamma = config.gamma  # discount factor for future rewards
        self.batch_size = config.batch_size  # in 4trajectories (episodes)
        self.optimizer = torch.optim.Adam(self.pi.parameters(),
                                          lr=config.learning_rate)
        # self.optimizer =
        # tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        self.use_entropies = use_entropies
        # Set random seed for reproducibility
        self.env.seed(config.seed)
        # tf.set_random_seed(config.seed)
        torch.manual_seed(config.seed)
        self.decay = decay
        if decay:
            self.decay_lr = optim.lr_scheduler\
                                 .StepLR(self.optimizer,
                                         step_size=int(1000 /
                                                       self.batch_size),
                                         gamma=0.3)
        self.pi.train()  # DONT FORGET TO PUT MODEL TO TRAIN ON INIT!!!!!!!

    def save(self, filename):
        """ Save policy weights to a ***.h5 file. """
        self.pi.save_state_dict(filename)
        # self.pi.save_weights(filename)

    def load(self, filename):
        """ Load policy weights from a ***.h5 file. """
        # self.pi.load_weights(filename)
        self.pi.load_state_dict(torch.load(filename))

    def select_action(self, state):
        probs = self.pi(Variable(state))
        action = probs.multinomial(num_samples=1).data
        prob = probs[:, action[0, 0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        return action[0], log_prob, entropy

    def play(self, render):
        """
        Run the trained policy on the env.
        Returns the rewards and the rendered frames for visualization.
        """
        # put model to eval
        self.pi.eval()
        state = torch.Tensor([self.env.reset()])
        done = False
        rews, frames = [], []
        while not done:
            if render:
                frames.append(self.env.render(mode='rgb_array'))
                # We need to convert ob to a tensor, and add a batch dimension
                # and remove it again once we are done with TF ops
            action, log_prob, entropy = self.select_action(state)
            next_state, rew, done, info = self.env.step(action.numpy()[0])
            state = torch.Tensor([next_state])
            rews.append(rew)
        return rews, frames

    def _discount_rewards(self, rews):
        """
        Compute discounted rewards recursively.
        Recall that with discount, r_t = sum_{k=t}^{T} gamma^{k-t} * r_k
        """
        discounted_rews = []
        last_r = 0.
        for r in reversed(rews):
            last_r = last_r * self.gamma + r
            discounted_rews.append(last_r)
        discounted_rews.reverse()
        return discounted_rews

    # def _compute_gradients_tf(self, obs, actions, advantages):
    #     """
    #     Implement the policy gradient in TF. In order to do so with automatic
    #     differentiation, we need to build a surrogate loss whose gradient is
    #     the policy gradient.
    #     This is achieved by taking the logprob of the selected action and
    #     scaling it by the discounted reward.
    #     """
    #     advantages = np.array(advantages)  # [batch_size]
    #     actions = np.array(actions)  # [batch_size]
    #     with tf.GradientTape() as tape:
    #         logprobs = self.pi(np.array(obs))  # [batch_size, num_actions]
    #         # Compute PG surrogate and  flip sign for gradient ascent
    #         policy_grad_surrogate = -tf.nn\
    #                                    .sparse_softmax_cross_entropy_with_logits(
    #                                        logits=logprobs, labels=actions)
    #         # Scale PG by the advantages
    #         print(policy_grad_surrogate)
    #         loss = tf.reduce_mean(advantages * policy_grad_surrogate)
    #     return tape.gradient(loss, self.pi.variables)

    def _compute_gradients_torch(self, rewards, log_probs, entropies):
        """
        Implement the policy gradient in Torch. In order to do so with
        automatic differentiation, we need to build a surrogate loss whose
        gradient is the policy gradient.
        This is achieved by taking the logprob of the selected action and
        scaling it by the discounted reward.
"""
        R = torch.zeros(1, 1)
        self.pi.train()  # ojuu que el posarem a val a la funcio play
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            if self.use_entropies:
                loss = loss - (log_probs[i]*(Variable(R)
                                             .expand_as(log_probs[i]))).sum() \
                       - (0.0001*entropies[i]).sum()
            else:
                loss = loss - (log_probs[i]*(Variable(R)
                                             .expand_as(log_probs[i]))).sum()

        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.pi.parameters(), 40*self.batch_size)
        self.optimizer.step()
        if self.decay:
            self.decay_lr.step()
        return rewards, loss

    def train(self, num_episodes):
        """
        Training loop:
          1. Collect experience by running the policy on the env
          2. Estimate the policy gradient with samples
          3. Update the policy and go back to (1)
        """
        rew_hist = []
        eval_cum_rews_history = []
        entropies = []
        log_probs = []
        losses = []
        rewards = []
        for i_episode in range(num_episodes):
            state = torch.Tensor([self.env.reset()])

            done = False
            while not done:  # PLAY PLAY
                action, log_prob, entropy = self.select_action(state)
                action = action

                next_state, reward, done, _ = self.env.step(action.numpy()[0])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = torch.Tensor([next_state])

                if done:
                    break
            # UPDATE GRADIENTSSSS AND VAL THE AGENT WITH ONE EPISODE
            if i_episode % self.batch_size == 0:
                ep_rew, loss = self._compute_gradients_torch(rewards,
                                                             log_probs,
                                                             entropies)

                rew_hist.append(ep_rew)
                eval_rews, _ = self.play(render=False)
                eval_cum_rew = np.sum(eval_rews)
                eval_cum_rews_history.append(eval_cum_rew)
                losses.append(loss)
                entropies = []
                log_probs = []
                rewards = []
                torch.save(self.pi.state_dict(), 'wpt.pkl')
                sys.stdout.write('\r%s %s %s %s %s %s' % ('Train rew avg: ',
                                                          np.array(ep_rew)
                                                          .sum()
                                                          / self.batch_size,
                                                          ' val: ',
                                                          eval_cum_rew,
                                                          '        ',
                                                          '              '))
                sys.stdout.flush()

        return eval_cum_rews_history, losses


class CartPoleConfig:
    env = "CartPole-v0"
    seed = 0  # for reproducibility0
    gamma = 0.99  # discount factor for future rewards
    batch_size = 64  # in trajectories (episodes)
    learning_rate = 0.0001


def ewma(x, alpha):
    """
    Exponential Weighted Moving Average.
    Source: https://stackoverflow.com/a/42905202
    """
    x = np.array(x)
    n = x.size
    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    p = np.vstack([np.arange(i, i-n, -1) for i in range(n)])
    w = np.tril(w0**p, 0)
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)


def main():
    entropic_agent = Agent(CartPoleConfig, use_entropies=True)
    agent = Agent(CartPoleConfig, use_entropies=False)

    n_experiments = 10
    entropic_rewards = []
    normal_rewards = []
    # Training on 3000 episodes takes around 3:30min on CPU
    for _i in range(n_experiments):
        print("Experiment: {}".format(_i+1))
        reward, loss = agent.train(num_episodes=20000)
        print(" Loss: {} Reward: {} \n".format(reward, loss))
        normal_rewards.append(reward)
        reward, loss = entropic_agent.train(num_episodes=20000)
        entropic_rewards.append(reward)
        # upss, dont forget to go back to dumb agent!!!
        entropic_agent = Agent(CartPoleConfig, use_entropies=True)
        agent = Agent(CartPoleConfig, use_entropies=False)
        print("Entropic Loss: {} Reward: {} \n".format(reward, loss))
        CartPoleConfig.seed += 1
        break

    # Plot the reward, with and without EWMA smoothing
    plt.figure(figsize=(20, 6))

    # avg rewards
    print(np.array(normal_rewards).shape)
    # Change name, if u run it twice u will loose ut rewards...
    normal_rewards_a = np.mean(np.array(normal_rewards), axis=0)
    entropic_rewards_a = np.mean(np.array(entropic_rewards), axis=0)
    axarr = plt.subplot(1, 2, 1)
    axarr.plot(ewma(normal_rewards_a, alpha=1.))
    axarr.plot(ewma(entropic_rewards_a, alpha=1.))

    axarr.set_title('(Raw) Cumulative Reward')
    axarr.set_ylabel('Cumulative reward')
    axarr.set_xlabel('SGD steps')

    axarr = plt.subplot(1, 2, 2)
    axarr.set_title('Smoothed Cumulative Reward')
    axarr.plot(ewma(normal_rewards_a, alpha=0.1))
    axarr.plot(ewma(entropic_rewards_a, alpha=0.1))

    axarr.set_ylabel('Cumulative reward')
    axarr.set_xlabel('SGD steps')

    plt.show()

    probs = torch.Tensor([[0.1, 0.9]])
    print(probs)

    action = probs.multinomial(num_samples=1).data
    print(action)

    prob = probs[:, action[0, 0]].view(1, -1)
    print(prob)

    log_prob = prob.log()
    print(log_prob)

    entropy = - (probs*probs.log()).sum()
    print(entropy)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("Action space (discrete): {}".format(env.action_space.n))
    print("Observation space (discrete): {}".format(
        env.observation_space))

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
