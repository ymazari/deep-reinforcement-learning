import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        if states.dim() == 1:
            states = states.unsqueeze(0)

        x = F.relu(self.bn1(self.fc1(states)))
        x = F.relu(self.bn2(self.fc2(x)))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        # Critic network maps (all_states, all_actions) pairs to Q-values
        if states.dim() == 1:
            states = states.unsqueeze(0)

        xs = F.relu(self.bn1(self.fc1(states)))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
