import torch.optim as optim


class Optimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class AdamOptimizer(Optimizer):
    def __init__(self, model, learning_rate):
        super(AdamOptimizer, self).__init__(optimizer=optim.Adam(model.parameters(), lr=learning_rate))

