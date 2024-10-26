import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import os
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Optional
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


class ReplayBufferActor(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
        )



def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device.type == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


class ReplayBufferAtari:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)


class ModelBasedActorReplayBuffer(object):
    def  __init__(self, state_dim, action_dim, horizon, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, horizon, state_dim))
        self.action = np.zeros((max_size, horizon, action_dim))
        self.next_state = np.zeros((max_size, horizon, state_dim))
        self.reward = np.zeros((max_size, horizon))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward):
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state

        self.action[self.ptr] = action
        self.reward[self.ptr] = reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
        )


class TAACReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.prev_action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, prev_action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.prev_action[self.ptr] = prev_action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.prev_action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def sample_sequence(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)


class TAACReplayBufferLatent(object):
    def __init__(self, state_dim, action_dim, horizon=5, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim * horizon))
        self.prev_action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim * horizon))
        self.reward = np.zeros((max_size, horizon))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, prev_action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.prev_action[self.ptr] = prev_action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.prev_action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class FiGARReplayBuffer(object):
    def __init__(self, state_dim, action_dim, rep_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.rep = np.zeros((max_size, rep_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, rep, next_state, next_action, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.rep[self.ptr] = rep
        self.next_state[self.ptr] = next_state
        self.next_action[self.ptr] = next_action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.rep[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class HyperReplayBuffer(object):
    def __init__(self, state_dim, action_dim, clock_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.clock = np.zeros((max_size, clock_dim))
        self.next_clock = np.zeros((max_size, clock_dim))
        self.hyper_state = np.zeros((max_size, state_dim))
        self.hyper_next_state = np.zeros((max_size, state_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, clock, next_clock, hyper_state, hyper_next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.clock[self.ptr] = clock
        self.next_clock[self.ptr] = next_clock
        self.hyper_state[self.ptr] = hyper_state
        self.hyper_next_state[self.ptr] = hyper_next_state
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.clock[ind]).to(self.device),
            torch.FloatTensor(self.next_clock[ind]).to(self.device),
            torch.FloatTensor(self.hyper_state[ind]).to(self.device),
            torch.FloatTensor(self.hyper_next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class PlanReplayBuffer(object):
    def __init__(self, state_dim, action_dim, plan_actions, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim * plan_actions))
        self.next_state = np.zeros((max_size, state_dim * plan_actions))
        self.reward = np.zeros((max_size, plan_actions))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class NashReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action_fast = np.zeros((max_size, action_dim))
        self.action_slow = np.zeros((max_size, action_dim))

        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.change_slow = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action_fast, action_slow, next_state, reward, done, change_slow):
        self.state[self.ptr] = state
        self.action_fast[self.ptr] = action_fast
        self.action_slow[self.ptr] = action_slow

        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.change_slow[self.ptr] = change_slow

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action_fast[ind]).to(self.device),
            torch.FloatTensor(self.action_slow[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.change_slow[ind]).to(self.device)
        )

class Reflex(nn.Module):
    def __init__(self):
        super(Reflex, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class HandCraftedReflex(nn.Module):
    def __init__(self, observation_space, threshold=0.15, reflex_force_scale=1.0):
        super(HandCraftedReflex, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space)
        self.reflex_detector = nn.Linear(input_dim, 2)
        self.reflex_detector.weight.requires_grad = False
        self.reflex_detector.bias.requires_grad = False
        self.reflex_detector.weight.data = torch.zeros(self.reflex_detector.weight.shape)
        self.reflex_detector.weight.data[0, 1] = 1
        self.reflex_detector.weight.data[1, 1] = -1
        self.reflex_detector.bias.data = torch.ones(self.reflex_detector.bias.data.shape) * threshold * -1

        self.reflex = nn.Linear(2, 1)
        self.reflex.weight.requires_grad = False
        self.reflex.bias.requires_grad = False
        self.reflex.weight.data[0, 0] = reflex_force_scale / (0.20 - threshold)
        self.reflex.weight.data[0, 1] = -reflex_force_scale / (0.20 - threshold)
        self.reflex.bias.data[0] = 0

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a1 = F.relu(self.reflex_detector(state))
        reflex = self.reflex(a1)
        return reflex


class CEMReflex(nn.Module):
    def __init__(self, observation_space, action_space, thresholds=None, reflex_force_scales=None):
        super(CEMReflex, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space)
        num_action = len(action_space)
        observation_dim = input_dim - num_action

        if thresholds is None:
            thresholds = np.zeros(observation_dim * num_action)
        if reflex_force_scales is None:
            reflex_force_scales = np.zeros(observation_dim * num_action)

        self.reflex_detector = nn.Linear(input_dim, observation_dim * 2 * num_action)
        self.reflex_detector.weight.requires_grad = False
        self.reflex_detector.bias.requires_grad = False
        self.reflex_detector.weight.data = torch.zeros(self.reflex_detector.weight.shape)
        for action in range(num_action):
            for i in range(observation_dim):
                self.reflex_detector.weight.data[(action * observation_dim * 2) + i * 2, i] = 1
                self.reflex_detector.weight.data[(action * observation_dim * 2) + i * 2 + 1, i] = -1
                self.reflex_detector.bias.data[(action * observation_dim * 2) + i * 2] = thresholds[(action * observation_dim) + i] * -1
                self.reflex_detector.bias.data[(action * observation_dim * 2) + i * 2 + 1] = thresholds[(action * observation_dim) + i] * -1

        self.reflex = nn.Linear(observation_dim * 2 * num_action, num_action)
        self.reflex.weight.requires_grad = False
        self.reflex.bias.requires_grad = False
        for action in range(num_action):
            for i in range(observation_dim):
                self.reflex.weight.data[action,  (action*observation_dim * 2) + (i * 2)] = reflex_force_scales[(action * observation_dim) + i]
                self.reflex.weight.data[action, (action*observation_dim * 2) + (i * 2) + 1] = -reflex_force_scales[(action * observation_dim) + i]
        self.reflex.bias.data = torch.zeros(self.reflex.bias.shape)

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a1 = F.relu(self.reflex_detector(state))
        reflex = self.reflex(a1)
        return reflex


class StatesDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        state = self.df['states'].iloc[idx]
        action = self.df['action'].iloc[idx][0]
        failure = self.df['failure'].iloc[idx]
        label = 0 if failure == 0.0 else action

        return torch.Tensor(state), label


def append_data_to_excel(excel_name, columns, data):
    if not os.path.isfile(os.path.join(excel_name)):
        with open(os.path.join(excel_name), 'w') as f:
            f.write(','.join([str(x) for x in columns]) + '\n')

    with open(os.path.join(excel_name), 'a') as f:
        f.write(','.join([str(x) for x in data]) + '\n')


class Clock:
    def __init__(self, clock_dim=2):
        self.state = np.zeros(clock_dim)
        self.value = 0
        self.clock_dim = clock_dim

    def tick(self):
        self.value += 1
        if self.value >= np.power(2, self.clock_dim):
            self.value = 0
        self.state = np.unpackbits(
            np.array([self.value], dtype='>i8').view(np.uint8))[-self.clock_dim:]
        return self.state

    def reset(self):
        self.value = 0
        self.state = np.unpackbits(
            np.array([self.value], dtype='>i8').view(np.uint8))[-self.clock_dim:]
        return self.state


class OneHotClock:
    def __init__(self, clock_dim=2):
        self.state = np.zeros(clock_dim)
        self.state[0] = 1
        self.clock_dim = clock_dim
        self.value = 0

    def tick(self):
        self.value += 1
        if self.value >= self.clock_dim:
            self.value = 0

        self.state = np.zeros(self.clock_dim)
        self.state[self.value] = 1
        return self.state

    def reset(self):
        self.value = 0
        self.state = np.zeros(self.clock_dim)
        self.state[0] = 1
        return self.state



class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
    ):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) *
            np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(
            self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"