from collections import deque, namedtuple
import random
import torch
import numpy as np
from model import DQNetwork
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """ Utility class to manage experience replays.
        Credit: Taken verbatim from Udacity examples. """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        :param action_size: The number of all legal actions
        :param buffer_size: The number of items the buffer can keep
        :param batch_size: The number of samples to replay from memory buffer
        :param seed: random seed
        """

        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """ Adds the experience to the memory buffer """
        an_experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(an_experience)

    def sample(self):
        """ randomly sample a batch of experiences from memory """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class YellowBananaThief:
    """ A smart agent that interacts with the environment to pick up yellow bananas"""

    def __init__(self, state_size, action_size, seed=0, buffer_size=100000, batch_size=64, update_frequency=2,
                 gamma=.99, learning_rate=5e-4, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.random = random.seed(seed)
        self.batch_size = batch_size

        self.memory = ReplayBuffer(self.action_size, buffer_size, batch_size, seed)
        self.time_step = 0
        self.update_frequency = update_frequency

        self.qnetwork_local = DQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau

    def act(self, state, epsilon):
        """ Returns an epsilon greedy action to take in the current state
            :param state: The current state in the environment
            :param epsilon: Epsilon value to apply epsilon-greedy action selection
        """
        def action_probabilities(action_vals, eps, num_actions):
            """ Determine the epsilon probabilities of choosing actions """
            probs = np.ones(num_actions, dtype=float) * (eps / num_actions)
            best_action = np.argmax(action_vals)
            probs[best_action] += (1. - eps)
            return probs

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # get the network in evaluation mode and pull values from it
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # get the network back into train mode

        action_probs = action_probabilities(action_values.cpu().data.numpy(), epsilon, self.action_size)
        return np.random.choice(np.arange(self.action_size), p=action_probs)

    def step(self, state, action, reward, next_state, done):
        """ Step forward to train the model """
        self.memory.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1) % self.update_frequency
        if self.time_step == 0:
            if len(self.memory) > self.batch_size:
                # enough samples have been collected for learning from experience
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """ Train the agent from a sample of experiences """
        def soft_update(local_model, target_model, tau):
            """Soft update model parameters.
                   θ_target = τ*θ_local + (1 - τ)*θ_target

                   Params
                   ======
                       local_model (PyTorch model): weights will be copied from
                       target_model (PyTorch model): weights will be copied to
                       tau (float): interpolation parameter
                   """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        states, actions, rewards, next_states, dones = experiences

        # max predicted Q values for the next state
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Q targets for current state
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # get expected q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # compute model loss
        loss = F.mse_loss(q_expected, q_targets)

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def local_qnet(self):
        """ Returns the trained model """
        return self.qnetwork_local
