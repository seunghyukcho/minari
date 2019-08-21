import torch
import torch.nn as nn
import torch.nn.functional as torch_function


class TwoLevelNetwork(nn.Module):
    def __init__(self, input_size1, input_size2):
        super(TwoLevelNetwork, self).__init__()

        self.f1 = nn.Linear(input_size1, 200)
        self.f2 = nn.Linear(200, 200)
        self.f3 = nn.Linear(200, 1)
        
        self.s1 = nn.Linear(input_size2 + 1, 100)
        self.s2 = nn.Linear(100, 100)
        self.s3 = nn.Linear(100, 1)

        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.xavier_uniform_(self.f3.weight)
        
        nn.init.xavier_uniform_(self.s1.weight)
        nn.init.xavier_uniform_(self.s2.weight)
        nn.init.xavier_uniform_(self.s3.weight)

    def forward(self, x1, x2):
        x1 = torch_function.leaky_relu(self.f1(x1))
        x1 = torch_function.leaky_relu(self.f2(x1))
        x1 = torch_function.leaky_relu(self.f3(x1))
        x2 = torch.cat((x1, x2), 1)
        
        x2 = torch_function.leaky_relu(self.s1(x2))
        x2 = torch_function.leaky_relu(self.s2(x2))
        x2 = torch_function.leaky_relu(self.s3(x2))

        return x2
