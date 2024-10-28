import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, state_dim, action_dim, max_action, neurons=[400, 300]):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        output = self.max_action * torch.tanh(self.l3(a))
        return output


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


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, neurons).to(device)
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


class TLA(TD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            neurons=[400, 300],
            lr=3e-4,
    ):
        super(TLA, self).__init__(state_dim, action_dim, max_action, discount, tau,
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

