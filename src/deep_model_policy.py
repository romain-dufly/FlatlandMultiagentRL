import copy
import os
import pickle
import random
from collections import namedtuple, deque
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class PolicyParameters:
    def __init__(self, action_size, buffer_size, batch_size, update_every, gamma, tau, lr, use_gpu):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = lr
        self.use_gpu = use_gpu

class Policy:
    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps=0.):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

class DeepPolicy(Policy):
    """
    Wrapper on a network that defines a policy for the agent
    Inputs: model (an extension of nn.Module)
            p (parameters in the form of a class)
            evaluation_mode (boolean)
    """
    def __init__(self, model, p, evaluation_mode=False):
        
        self.evaluation_mode = evaluation_mode
        self.action_size = p.action_size

        if p.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model = model.to(self.device)

        if not evaluation_mode:
            self.target = copy.deepcopy(self.model) # target network
            self.optimizer = optim.Adam(self.model.parameters(), lr=p.learning_rate)
            self.memory = ReplayBuffer(p.action_size, p.buffer_size, p.batch_size, self.device)
            self.batch_size = p.batch_size
            self.gamma = p.gamma # discount factor
            self.tau = p.tau # for soft update of target parameters
            self.update_every = p.update_every # interval for updating the network
            self.t_step = 0 # counter for learning steps

    def act(self, state, eps=0.):
        self.model.eval()
        with torch.no_grad():
            if isinstance(state, tuple):
                action_values = self.model(*state)
            else:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_values = self.model(state)
        self.model.train()

        # Use an epsilon-greedy policy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def act_centralized(self, state, eps=0.):
        self.model.eval()
        with torch.no_grad():
            if isinstance(state, tuple):
                action_values = self.model(*state)
            else:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_values = self.model(state)
        self.model.train()

        if random.random() > eps:
            return [np.argmax(agent_values) for agent_values in action_values.cpu().data.numpy()[0]]
        else:
            return [random.choice(np.arange(self.action_size)) for _ in range(len(action_values.cpu().data.numpy()[0]))]

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy set to evaluation only."

        # Add to memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn at regular intervals
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                if isinstance(state, tuple):
                    experiences = self.memory.sample_lstm()
                    self.learn(experiences, 'LSTM')
                elif len(state.shape) >= 2:
                    experiences = self.memory.sample_centralized()
                    self.learn(experiences, 'centralized')
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def learn(self, experiences, special=None):
        """Update model parameters using given batch of experience."""
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        if special == 'LSTM':
            # For each element in the list of states, get the Q values efficiently and stack them
            Q_expected = torch.stack([self.model(*s) for s in states], dim=0).squeeze(1).squeeze(1)
            Q_expected = Q_expected.gather(1, actions)
        elif special == 'centralized':
            Q_expected = self.model(states).gather(2, actions.unsqueeze(-1)).squeeze(2)
        else:
            Q_expected = self.model(states).gather(1, actions)

        # Compute Q targets for current states
        if special == 'LSTM':
            Q_best_action = torch.stack([self.model(*s) for s in next_states], dim=0).squeeze(1).squeeze(1).max(1)[1]
            Q_targets_next = torch.stack([self.target(*s) for s in next_states], dim=0).squeeze(1).squeeze(1)
            Q_targets_next = Q_targets_next.gather(1, Q_best_action.unsqueeze(-1))
        elif special == 'centralized':
            Q_best_action = self.model(next_states).max(2)[1]
            Q_targets_next = self.target(next_states).gather(2, Q_best_action.unsqueeze(-1)).squeeze(2)
            dones = dones[:,:-1]
        else:
            Q_best_action = self.model(next_states).max(1)[1]
            Q_targets_next = self.target(next_states).gather(1, Q_best_action.unsqueeze(-1))
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for target_param, model_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * model_param.data + (1.0 - self.tau) * target_param.data)
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target = copy.deepcopy(self.model)
    
    def save_replay_buffer(self, filename, size):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-size:], f)
    
    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Inputs: action_size (int): dimension of each action
                buffer_size (int): maximum size of buffer
                batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # double ended queue
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if isinstance(state, tuple) or len(state.shape) >= 2:
            e = Experience(state, action, reward, next_state, done)
        else:
            e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones
    
    def sample_lstm(self):
        """Randomly sample a batch of experiences from memory when using LSTM."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [e.state for e in experiences if e is not None]
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = [e.next_state for e in experiences if e is not None]
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones
    
    def sample_centralized(self):
        """Randomly sample a batch of experiences from memory when using a centralized critic."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([[v for v in e.action.values()] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([[v for v in e.reward.values()] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([[v for v in e.done.values()] for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states