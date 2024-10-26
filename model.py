import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Mapping
from utils import Clock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, input_dim, skip_dim, non_linearity=F.relu):
        super(Q, self).__init__()
        # We follow the architecture of the Actor and Critic networks in terms of depth and hidden units
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, skip_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, neurons=[400, 300], gated=False):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], action_dim)

        self.max_action = max_action
        self.gated = gated

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        output = self.max_action * torch.tanh(self.l3(a))
        if self.gated:
            output[:, :-1] = output[:, :-1] * (output[:, -1:] > 0)
        return output


class HyperActor(nn.Module):
    def __init__(self, state_dim, action_dim, steps, clock_dim, max_action, neurons1=[400, 300], neurons2=5):
        super(HyperActor, self).__init__()

        self.clock_dim = clock_dim
        self.action_dim = action_dim
        self.steps = steps
        self.neurons = neurons2
        self.l1 = nn.Linear(state_dim, neurons1[0])
        self.l2 = nn.Linear(neurons1[0], neurons1[1])

        self.w1_size = clock_dim * neurons2
        self.w2_size = neurons2 * action_dim
        self.b1_size = neurons2
        self.b2_size = action_dim

        self.l3 = nn.Linear(neurons1[1], self.w1_size + self.w2_size + self.b1_size + self.b2_size)

        # self.l1.weight = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l1.weight).cuda(),2))
        # self.l2.weight = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l2.weight).cuda(),2))
        # self.l3.weight = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l3.weight).cuda(),2))
        # self.l4.weight = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l4.weight).cuda(),2))
        # self.l5.weight = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l5.weight).cuda(),2))
        # self.l6.weight = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l6.weight).cuda(),2))
        #
        # self.l1.bias = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l1.bias).cuda(), 2))
        # self.l2.bias = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l2.bias).cuda(), 2))
        # self.l3.bias = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l3.bias).cuda(), 2))
        # self.l4.bias = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l4.bias).cuda(), 2))
        # self.l5.bias = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l5.bias).cuda(), 2))
        # self.l6.bias = torch.nn.parameter.Parameter(torch.fmod(torch.randn_like(self.l6.bias).cuda(), 2))

        self.max_action = max_action

    def forward_state(self, state):
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))
        out = self.l3(hidden)
        out = torch.tanh(out)
        return out

    def forward(self, state, clock):
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))

        params = self.l3(hidden)
        params = torch.tanh(params)
        w1 = params[:, 0:self.w1_size].reshape([state.shape[0], self.clock_dim, self.neurons])
        w2 = params[:, self.w1_size:self.w1_size + self.w2_size].reshape(
            [state.shape[0], self.neurons, self.action_dim])

        b1 = params[:, self.w1_size + self.w2_size:self.w1_size + self.w2_size + self.b1_size].reshape(
            [state.shape[0], 1, -1])
        b2 = params[:,
             self.w1_size + self.w2_size + self.b1_size:self.w1_size + self.w2_size + self.b1_size + self.b2_size].reshape(
            [state.shape[0], 1, -1])

        clock = torch.from_numpy(clock.reshape([clock.shape[0], 1, clock.shape[1]])).to(torch.float32).to(device)
        a = F.relu(torch.matmul(clock, w1) + b1)

        a = torch.matmul(a, w2) + b2

        a = a.reshape(a.shape[0], a.shape[-1])
        output = self.max_action * torch.tanh(a)
        return output

    def forward_clock(self, clock, params):
        w1 = params[:, 0:self.w1_size].reshape([params.shape[0], self.clock_dim, self.neurons])
        w2 = params[:, self.w1_size:self.w1_size + self.w2_size].reshape(
            [params.shape[0], self.neurons, self.action_dim])

        b1 = params[:, self.w1_size + self.w2_size:self.w1_size + self.w2_size + self.b1_size].reshape(
            [params.shape[0], 1, -1])
        b2 = params[:,
             self.w1_size + self.w2_size + self.b1_size:self.w1_size + self.w2_size + self.b1_size + self.b2_size].reshape(
            [params.shape[0], 1, -1])
        clock = clock.reshape([clock.shape[0], 1, clock.shape[1]])
        a = F.relu(torch.matmul(clock, w1) + b1)
        a = torch.matmul(a, w2) + b2
        output = self.max_action * torch.tanh(a)
        return output


class GRUActor(nn.Module):
    def __init__(self, input_dim, action_dim, max_action):
        super(GRUActor, self).__init__()
        self.l1 = nn.Linear(input_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.gru = nn.GRUCell(action_dim, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state, previous_action, planning_horizon=1):
        a = F.relu(self.l1(state))
        a = self.l2(a)
        output_actions = list()
        for i in range(planning_horizon):
            a = self.gru(previous_action, a)
            output = self.max_action * torch.tanh(self.l3(a))
            output_actions.append(output)
            previous_action = output

        if planning_horizon > 1:
            return torch.stack(output_actions, dim=1)
        else:
            return output_actions[0]


class DelayedActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, neurons=[400, 300]):
        super(DelayedActor, self).__init__()

        self.l1 = nn.Linear(state_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class DelayedActorFastHybrid(nn.Module):
    def __init__(self, observation_space, action_dim, max_action):
        super(DelayedActorFastHybrid, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space) * 2 + action_dim
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class DelayedQuickActor(nn.Module):
    def __init__(self, observation_space, action_dim, max_action, threshold=0.15, reflex_force_scale=1.0):
        super(DelayedQuickActor, self).__init__()

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

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a1 = F.relu(self.reflex_detector(state))
        reflex = self.reflex(a1)

        a2 = F.relu(self.l1(state))
        # a = F.relu(self.l2(torch.cat((a2, a1), dim=1)))
        a = F.relu(self.l2(a2))

        return reflex, self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, neurons=[400, 300]):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l5 = nn.Linear(neurons[0], neurons[1])
        self.l6 = nn.Linear(neurons[1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, neurons=[400, 300]):
        super(Model, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], state_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DelayedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, neurons=[400, 300]):
        super(DelayedCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l5 = nn.Linear(neurons[0], neurons[1])
        self.l6 = nn.Linear(neurons[1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DelayedCriticFastHybrid(nn.Module):
    def __init__(self, observation_space, action_dim):
        super(DelayedCriticFastHybrid, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space) * 2 + action_dim
        # Q1 architecture
        self.l1 = nn.Linear(input_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            delayed_env=False,
            reflex=False,
            threshold=0.15,
            reflex_force_scale=1.0,
            fast_hybrid=False,
            neurons=[400, 300],
            lr=3e-4,
            gated=False
    ):

        self.delayed_env = delayed_env
        self.reflex = reflex
        self.gated = gated

        if reflex:
            self.actor = DelayedQuickActor(observation_space, action_dim, max_action, threshold, reflex_force_scale).to(
                device)
            self.critic = DelayedCritic(observation_space, action_dim).to(device)
        elif self.delayed_env:
            if fast_hybrid:
                self.actor = DelayedActorFastHybrid(observation_space, action_dim, max_action).to(device)
                self.critic = DelayedCriticFastHybrid(observation_space, action_dim).to(device)
            else:
                self.actor = DelayedActor(state_dim, action_dim, max_action, neurons).to(device)
                self.critic = DelayedCritic(state_dim, action_dim, neurons).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action, neurons, gated).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if self.reflex:
            return self.actor(state)[0].cpu().data.numpy().flatten(), self.actor(state)[1].cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            if self.reflex:
                next_action = (
                        self.actor_target(next_state)[1] + noise
                ).clamp(-self.max_action, self.max_action)
            else:
                next_action = (
                        self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            if self.reflex:
                actor_loss = -self.critic.Q1(state, self.actor(state)[1]).mean()
            else:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class TD3Discount(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, neurons, False).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic_target2 = copy.deepcopy(self.critic2)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target1_Q1, target1_Q2 = self.critic_target1(next_state, next_action)
            target1_Q = torch.min(target1_Q1, target1_Q2)
            target1_Q = reward + not_done * self.discount * target1_Q

            target2_Q1, target2_Q2 = self.critic_target2(next_state, next_action)
            target2_Q = torch.min(target2_Q1, target2_Q2)
            target2_Q = reward + not_done * 0.999 * target2_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic1(state, action)

        # Compute critic loss
        critic_loss1 = F.mse_loss(current_Q1, target1_Q) + F.mse_loss(current_Q2, target1_Q)

        # Optimize the critic
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()
        cl1 = critic_loss1.item()

        current_Q1, current_Q2 = self.critic2(state, action)

        # Compute critic loss
        critic_loss2 = F.mse_loss(current_Q1, target2_Q) + F.mse_loss(current_Q2, target2_Q)

        # Optimize the critic
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()
        cl2 = critic_loss2.item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            critic_val1 = self.critic1.Q1(state, self.actor(state))
            critic_val2 = self.critic2.Q1(state, self.actor(state))
            total_loss = cl1 + cl2
            actor_loss = (- (critic_val1 * cl2 / total_loss) - (critic_val2 * cl1 / total_loss)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss1

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic_optimizer1.state_dict(), filename + "_critic_optimizer1")

        torch.save(self.critic2.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer2.state_dict(), filename + "_critic_optimizer2")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic_optimizer1.load_state_dict(torch.load(filename + "_critic_optimizer1"))
        self.critic_target1 = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic_optimizer2.load_state_dict(torch.load(filename + "_critic_optimizer2"))
        self.critic_target2 = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class POTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):
        self.removed_obs = observation_space.shape[0] - state_dim
        self.actor = Actor(state_dim, action_dim, max_action, neurons).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, critic, batch_size=256, ):
        self.total_it += 1

        # Sample replay buffer

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            # Compute actor loss

            actor_loss = -critic.Q1(state, self.actor(state[:, self.removed_obs:])).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))


class TempoRLPOTLA(POTD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
            skip_dim=2
    ):
        super(TempoRLPOTLA, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                           policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr)
        self.skip_Q = Q(state_dim + action_dim, skip_dim).to(device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, critic, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, next_action, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            target_Q = critic.Q1(next_state, self.actor_target(next_state[:, self.removed_obs:]))
            target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state[:, self.removed_obs:], action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


class PlanTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
            plan_actions=1,
    ):
        self.plan_actions = plan_actions
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor = Actor(state_dim, action_dim * plan_actions, max_action, neurons).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, critic, batch_size=256):
        self.total_it += 1

        # with torch.no_grad():
        #     # Select action according to policy and add clipped noise
        #
        #     noise = (
        #             torch.randn_like(action[:self.action_dim]) * self.policy_noise
        #     ).clamp(-self.noise_clip, self.noise_clip)
        #     # next_action = (
        #     #         self.actor_target(next_states[:, self.state_dim * (self.plan_actions-1):]) + noise
        #     # ).clamp(-self.max_action, self.max_action)[:, :self.action_dim]
        #     next_action = (
        #             self.actor_target(next_states[:, :self.state_dim]) + noise
        #     ).clamp(-self.max_action, self.max_action)[:, :self.action_dim]
        #
        #     # Compute the target Q value
        #     # target_Q1, target_Q2 = self.critic_target(next_states[:, self.state_dim * (self.plan_actions-1):], next_action)
        #     target_Q1, target_Q2 = self.critic_target(next_states[:, :self.state_dim], next_action)
        #
        #     target_Q = torch.min(target_Q1, target_Q2)
        #
        #     # target_Q = reward[:, -1:] + not_done * self.discount * target_Q
        #     target_Q = reward[:, :1] + not_done * self.discount * target_Q
        #
        # # Get current Q estimates
        # # current_Q1, current_Q2 = self.critic(next_states[:, self.state_dim * (self.plan_actions-2):self.state_dim * (self.plan_actions-1)], action[:, self.action_dim*(self.plan_actions-1):])
        # current_Q1, current_Q2 = self.critic(state, action[:, :self.action_dim])
        #
        #
        # # Compute critic loss
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #
        # # Optimize the critic
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # Delayed policy updates

        if self.total_it % self.policy_freq == 0:
            self.critic = copy.deepcopy(critic)
            # Sample replay buffer
            state, action, next_states, reward, not_done = replay_buffer.sample(batch_size)

            # Compute actor loss
            actor_loss = 0
            actions = self.actor(state)
            for s in range(self.plan_actions):
                if s == 0:
                    actor_loss += (-self.critic.Q1(state, actions[:,
                                                          self.action_dim * s:self.action_dim * (s + 1)]).mean()) * (
                                              1 / ((s + 1) ** 2))
                else:
                    actor_loss += (-self.critic.Q1(next_states[:, self.state_dim * (s - 1):self.state_dim * s],
                                                   actions[:,
                                                   self.action_dim * s:self.action_dim * (s + 1)]).mean()) * (
                                              1 / ((s + 1) ** 2))

            # actor_loss = actor_loss/self.plan_actions
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class TempoRLPlanTD3(PlanTD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
            plan_actions=1
    ):
        super(TempoRLPlanTD3, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                             policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr,
                                             plan_actions=plan_actions)
        self.skip_Q = Q(state_dim + action_dim * plan_actions, 2).to(device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())
        self.action_dim = action_dim

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic.Q1(next_state, self.actor_target(next_state)[:, :self.action_dim])
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


class TempoRLTD3(TD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
            skip_dim=1
    ):
        super(TempoRLTD3, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                         policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr)
        self.skip_Q = Q(state_dim + action_dim, skip_dim).to(device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, _, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic.Q1(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * torch.pow(self.discount, skip + 1) * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


class TempoRLTLA(TD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):
        super(TempoRLTLA, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                         policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr)
        self.skip_Q = Q(state_dim + action_dim, 2).to(device)
        # self.skip_Q_target = copy.deepcopy(self.skip_Q)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters(), lr=lr)

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, batch_size=256):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, _, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target.Q1(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


class TempoRLTLAPreGate(TD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):
        super(TempoRLTLAPreGate, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                                policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr)
        self.skip_Q = Q(state_dim, 2).to(device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.skip_Q(state).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic.Q1(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(state).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


class SwitchTLA(TD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4
    ):
        super(SwitchTLA, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                        policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr)
        self.skip_Q = Q(state_dim, 0, 2).to(device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

    def select_skip(self, state):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.skip_Q(state).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic.Q1(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(state).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


class TAACTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, neurons, False).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, prev_action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            with torch.no_grad():
                beta_0 = self.critic.Q1(next_state, action)
            beta_1 = self.critic.Q1(next_state, next_action)
            logits = torch.cat((beta_0, beta_1), dim=1)
            b = torch.distributions.Categorical(logits=logits).sample().reshape(-1, 1)

            next_action = (next_action * b) + (1 - b) * action

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_action = self.actor(state)
            with torch.no_grad():
                beta_0 = self.critic.Q1(next_state, action)
            beta_1 = self.critic.Q1(state, actor_action)
            logits = torch.cat((beta_0, beta_1), dim=1)
            b = torch.distributions.Categorical(logits=logits).sample().reshape(-1, 1)
            a = (actor_action * b) + (1 - b) * prev_action

            actor_loss = -self.critic.Q1(state, a).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class NashTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            delayed_env=False,
            neurons_fast=[400, 300],
            neurons_slow=[400, 300],
            lr=1e-4
    ):

        self.delayed_env = delayed_env

        if self.delayed_env:
            self.actor_fast = DelayedActor(state_dim, action_dim, max_action, neurons_fast).to(device)
            self.actor_slow = DelayedActor(state_dim, action_dim, max_action, neurons_slow).to(device)
            self.critic = DelayedCritic(state_dim, action_dim).to(device)
        else:
            self.actor_fast = Actor(state_dim, action_dim, max_action, neurons_fast).to(device)
            self.actor_slow = Actor(state_dim, action_dim, max_action, neurons_slow).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_fast_target = copy.deepcopy(self.actor_fast)
        self.actor_slow_target = copy.deepcopy(self.actor_slow)

        self.actor_optimizer = torch.optim.Adam([{'params': self.actor_slow.parameters()},
                                                 {'params': self.actor_fast.parameters()}], lr=lr)
        # self.actor_slow_optimizer = torch.optim.Adam(self.actor_slow.parameters(), lr=lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor_fast(state).cpu().data.numpy().flatten(), self.actor_slow(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action_fast, action_slow, next_state, reward, not_done, change_slow = replay_buffer.sample(batch_size)
        action = (action_fast + action_slow).clamp(-self.max_action, self.max_action)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action_fast) * self.policy_noise / 2
            ).clamp(-self.noise_clip / 2, self.noise_clip / 2)

            next_action_fast = (
                    self.actor_fast_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            next_action_slow = (
                    self.actor_slow_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            next_action_slow = (next_action_slow * (change_slow)) + (action_slow * (1 - change_slow))

            next_action = (next_action_fast + next_action_slow).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            fast_action = self.actor_fast(state)
            slow_action = self.actor_slow(state)
            # slow_action = (change_slow * slow_action) + ((1-change_slow) * action_slow)

            action = (fast_action + slow_action)
            actor_loss = -self.critic.Q1(state, action).mean()

            # Optimize the actor
            # self.actor_fast_optimizer.zero_grad()
            # self.actor_slow_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.actor_fast_optimizer.step()
            # self.actor_slow_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_slow.parameters(), self.actor_slow_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_fast.parameters(), self.actor_fast_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor_fast.state_dict(), filename + "_actor_fast")
        # torch.save(self.actor_fast_optimizer.state_dict(), filename + "_actor_fast_optimizer")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_fast_optimizer")

        torch.save(self.actor_slow.state_dict(), filename + "_actor_slow")
        # torch.save(self.actor_slow_optimizer.state_dict(), filename + "_actor_slow_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_fast.load_state_dict(torch.load(filename + "_actor_fast"))
        self.actor_fast_optimizer.load_state_dict(torch.load(filename + "_actor_fast_optimizer"))
        self.actor_fast_target = copy.deepcopy(self.actor_fast)

        self.actor_slow.load_state_dict(torch.load(filename + "_actor_slow"))
        self.actor_slow_optimizer.load_state_dict(torch.load(filename + "_actor_slow_optimizer"))
        self.actor_slow_target = copy.deepcopy(self.actor_slow)


class EasyHyperTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            neurons2=5,
            lr=3e-4,
            clock_dim=5,
            hyperlr=2e-7,
            steps=1
    ):

        self.actor = HyperActor(state_dim, action_dim, steps, clock_dim, max_action, neurons, neurons2).to(device)
        self.critic = Critic(state_dim, (clock_dim * neurons2 + neurons2 + neurons2 * action_dim + action_dim),
                             neurons).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hyperlr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim

        self.total_it = 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.forward_state(next_state) + noise)
            # next_action = (self.actor(clock, next_w) + noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor.forward_state(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class HyperTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            neurons2=5,
            lr=3e-4,
            clock_dim=5,
            steps=2,
            hyperlr=3e-7
    ):
        self.clock_dim = clock_dim
        self.steps = steps
        self.actor = HyperActor(state_dim, action_dim, steps, clock_dim, max_action, neurons, neurons2).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.model = Model(state_dim, action_dim).to(device)

        self.model_target = copy.deepcopy(self.model)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hyperlr)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hyperlr)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model_loss_fn = nn.MSELoss()
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.total_it = 0

    def train_model(self, replay_buffer, batch_size):
        state, action, next_state, _, _ = replay_buffer.sample(batch_size)
        pred = self.model.forward(state, action)
        model_loss = self.model_loss_fn(pred, next_state)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        return model_loss

    def train_critic(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            clock = Clock(self.clock_dim)

            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.forward(next_state, np.expand_dims(clock.state, axis=0).repeat(batch_size,
                                                                                                            axis=0)) + noise)
            # next_action = (self.actor(clock, next_w) + noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def train(self, replay_buffer, critic, batch_size=256):
        self.total_it += 1
        self.critic = copy.deepcopy(critic)
        # Sample replay buffer
        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Sample replay buffer
            state = replay_buffer.sample(batch_size)[0]
            # Compute actor loss
            actor_loss = 0
            params = self.actor.forward_state(state)
            clock = Clock(self.clock_dim)
            for s in range(self.steps):
                action = self.actor.forward_clock(
                    torch.from_numpy(np.expand_dims(clock.state, axis=0).repeat(batch_size, axis=0)).float().to(device),
                    params).squeeze(1)
                actor_loss += torch.mean(-self.critic.Q1(state, action))
                next_state = self.model_target(state, action)
                state = next_state
                clock.tick()

            actor_loss = actor_loss / self.steps
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.model.state_dict(), filename + "_model")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        # self.model.load_state_dict(self.model.state_dict(), filename + "_model")
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class GRUTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=3e-4,
            actor_lr=3e-4,
            steps=2,
    ):

        self.steps = steps
        self.actor = GRUActor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.model = Model(state_dim, action_dim).to(device)

        self.model_target = copy.deepcopy(self.model)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model_loss_fn = nn.MSELoss()
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.total_it = 0

    def train_model(self, replay_buffer, batch_size):
        state, action, next_state, _, _ = replay_buffer.sample(batch_size)
        pred = self.model.forward(state, action)
        model_loss = self.model_loss_fn(pred, next_state)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        return model_loss

    def train_critic(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target.forward(next_state, action, 1) + noise
                           ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Sample replay buffer
            state, previous_action = replay_buffer.sample(batch_size)
            # Compute actor loss
            actor_loss = 0
            actions = self.actor.forward(state, previous_action, self.steps).float().squeeze(1)

            for ps in range(self.steps):
                actor_loss += -self.critic.Q1(state, actions[:, ps, :]).mean()
                next_state = self.model_target(state, actions[:, ps, :])
                state = next_state

            actor_loss = actor_loss / self.steps
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.model.state_dict(), filename + "_model")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        # self.model.load_state_dict(self.model.state_dict(), filename + "_model")
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class HyperPlanTD3(HyperTD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            neurons2=5,
            lr=3e-4,
            clock_dim=5,
            hyperlr=3e-7,
            independent_value=False,
            steps=2
    ):
        super(HyperPlanTD3, self).__init__(state_dim, action_dim, max_action, observation_space, discount, tau,
                                           policy_noise, noise_clip, policy_freq, neurons=neurons, lr=lr,
                                           neurons2=neurons2, clock_dim=clock_dim, hyperlr=hyperlr, steps=steps)
        self.skip_Q = Q(state_dim + (clock_dim * neurons2 + neurons2 + neurons2 * action_dim + action_dim), 2).to(
            device)
        self.skip_Q_target = copy.deepcopy(self.skip_Q)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

        if independent_value:
            self.train_skip = self.train_skip_independent
        else:
            self.train_skip = self.train_skip_joint_value

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = action.reshape(1, -1)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()

    def train_skip_independent(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, next_action, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            target_Q = \
            torch.max(self.skip_Q_target(torch.cat([next_state, self.actor_target.forward_state(next_state)], 1)), 1,
                      keepdim=True)[0]
            target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Update the frozen target models
            for param, target_param in zip(self.skip_Q.parameters(), self.skip_Q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss

    def train_skip_joint_value(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, _, reward, not_done = replay_buffer.sample(batch_size)
        clock = Clock(self.clock_dim)
        # Compute the target Q value
        with torch.no_grad():
            target_Q = self.critic.Q1(next_state, self.actor_target.forward(next_state,
                                                                            np.expand_dims(clock.state, axis=0).repeat(
                                                                                batch_size, axis=0)))
            target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

        return critic_loss

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))


def collate(batch, device=None):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch).to(device)
        # if elem.numel() < 20000:  # TODO: link to the relavant profiling that lead to this threshold
        #   return torch.stack(batch).to(device)
        # else:
        #   return torch.stack([b.contiguous().to(device) for b in batch], 0)
    elif isinstance(elem, np.ndarray):
        return collate(tuple(torch.from_numpy(b) for b in batch), device)
    elif hasattr(elem, '__torch_tensor__'):
        return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return type(elem)(collate(samples, device) for samples in transposed)
    elif isinstance(elem, Mapping):
        return type(elem)((key, collate(tuple(d[key] for d in batch), device)) for key in elem)
    else:
        return torch.from_numpy(np.array(batch)).to(device)

