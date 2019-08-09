import torch
import torch.nn as nn
import torch.nn.functional as torch_function


class TwoLevelNetwork(nn.Module):
    def __init__(self, input_size1, input_size2):
        super(TwoLevelNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size1, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 1)
        
        self.fc6 = nn.Linear(input_size2 + 1, 400)
        self.fc7 = nn.Linear(400, 400)
        self.fc8 = nn.Linear(400, 400)
        self.fc9 = nn.Linear(400, 400)
        self.fc10 = nn.Linear(400, 1)

    def forward(self, x1, x2):
        x1 = torch_function.leaky_relu(self.fc1(x1))
        x1 = torch_function.leaky_relu(self.fc2(x1))
        x1 = torch_function.leaky_relu(self.fc3(x1))
        x1 = torch_function.leaky_relu(self.fc4(x1))
        x1 = torch_function.leaky_relu(self.fc5(x1))
        x2 = torch.cat((x1, x2), 1)
        
        x2 = torch_function.leaky_relu(self.fc6(x2))
        x2 = torch_function.leaky_relu(self.fc7(x2))
        x2 = torch_function.leaky_relu(self.fc8(x2))
        x2 = torch_function.leaky_relu(self.fc9(x2))
        x2 = torch_function.leaky_relu(self.fc10(x2))

        return x2
