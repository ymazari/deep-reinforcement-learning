import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, 2 * state_size)
        self.fc2 = nn.Linear(2 * state_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        hidden = F.relu(self.fc1(state))
        output = self.fc2(hidden)
        return output
