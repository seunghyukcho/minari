import torch.nn as nn
import torch.nn.functional as torch_function


class SimpleNetwork(nn.Module):
    def __init__(self, input_size):
        super(SimpleNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch_function.leaky_relu(self.fc1(x))
        x = torch_function.leaky_relu(self.fc2(x))

        return x
