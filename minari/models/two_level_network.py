import torch
import torch.nn as nn
import torch.nn.functional as torch_function


class TwoLevelNetwork(nn.Module):
    def __init__(self, input_size1, input_size2):
        super(TwoLevelNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size1, 1)
        self.fc2 = nn.Linear(input_size2 + 1, 1)

    def forward(self, x1, x2):
        x1 = torch_function.leaky_relu(self.fc1(x1))
        x2 = torch.cat((x1, x2), 1)
        x2 = torch_function.leaky_relu(self.fc2(x2))

        return x2
