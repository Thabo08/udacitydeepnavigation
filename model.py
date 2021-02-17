import torch
import torch.nn as nn
import torch.nn.functional as F

    
class DQNetwork(nn.Module):
    """ The model to train the network """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, drop_out=.15):
        """
        Initialise model parameters
        :param state_size: Size of the state
        :param action_size: Size of the actions
        :param seed: Random seed
        :param fc1_units: Number of neurons in the first fully connected layer
        :param fc2_units: Number of neurons in the second fully connected layer
        :param drop_out: Dropout configuration indicating by how much drop out should be applied
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, state):
        """ Feed the input state through to the network in a forward step
            :param state: the current environment state to be fed through
        """
        x = F.relu(self.fc1(state))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        return F.relu(self.fc3(x))
