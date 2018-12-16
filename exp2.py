#!/usr/bin/python3
import gym
import math
import random
import os
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Some stuff that we will need to visualize the simulations in the notebook
import matplotlib.pyplot as plt

# Start virtual display
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)

# PARAMETERS and globals
screen_width = 600
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Config(object):
    """A Configuraiton object to set the paramet

    """
    def __init__(self, env, gamma, batch_size, seed=1):
        super(Config, self).__init__()
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.seed = seed


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent(object):
    """Implement the reinforcement learning algorithm

    """
    def __init__(self, config, device):
        super(Agent, self).__init__()
        # save the configuration in local attributes
        self.env = config.env
        self.env.seed = config.seed
        torch.manual_seed(config.seed)
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.device = device

        # create the Q-networks
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # create an optimizer (RMSprop)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        # create a memory
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []

    # TODO: dumb implementation
    def save(self, filepath):
        pass

    # TODO: dumb implementation
    def load(self, filepath):
        return Agent(None)

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        # MIDDLE OF CART
        return int(self.env.state[0] * scale + screen_width / 2.0)

    def get_screen(self, mode='rgb_array'):
        screen = self.env.render(mode=mode).transpose(
            (2, 0, 1))  # transpose into torch order (CHW)
        return screen

    def strip(self, screen):
        """
Strip off the edges, so that we have a square image centered on a cart
"""
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        screen = screen[:, :, slice_range]
        # Convert to float, rescare, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device,
                                dtype=torch.long)

    def plot_durations(self):
        # plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch)\
                                  .gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self\
                                            .target_net(non_final_next_states)\
                                            .max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) +\
                                       reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_state(self):
        self.last_screen = self.current_screen
        self.current_screen = self.strip(self.get_screen())
        return self.current_screen - self.last_screen

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            self.current_screen = self.strip(self.get_screen())
            state = self.get_state()
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if not done:
                    next_state = self.get_state()
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    break
            # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete')


def main():
    config = Config(gym.make('CartPole-v0').unwrapped,
                    GAMMA,
                    BATCH_SIZE)
    print("Action space (discrete): {}".format(config.env.action_space.n))
    print("Observation space (discrete): ", config.env.observation_space)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(config, device)
    # plt.figure()
    # plt.imshow(agent.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #            interpolation='none')
    # plt.title('Example extracted screen')

    num_episodes = 3500
    agent.train(num_episodes)
    agent.plot_durations()
    agent.env.render()
    agent.env.close()
    plt.ioff()


if __name__ == '__main__':
    main()
