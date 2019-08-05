import torch


class Loss(object):
    def __init__(self, criterion):
        self.criterion = criterion
        self.loss = None

    def compute_loss(self, predictions, targets):
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if not self.loss:
            raise ValueError("any loss isn't set yet...")
        self.loss.backward()


class RMSELoss(Loss):
    def __init__(self):
        super(RMSELoss, self).__init__(criterion=torch.nn.MSELoss())

    def compute_loss(self, predictions, targets):
        self.loss = torch.sqrt(self.criterion(predictions, targets))
        return self.loss


class MAELoss(Loss):
    def __init__(self):
        super(MAELoss, self).__init__(criterion=torch.nn.L1Loss())

    def compute_loss(self, predictions, targets):
        self.loss = self.criterion(predictions, targets)
        return self.loss


class NMAELoss(Loss):
    def __init__(self):
        super(NMAELoss, self).__init__(criterion=torch.nn.L1Loss())

    def compute_loss(self, predictions, targets):
        max_rate = torch.max(targets).item()
        min_rate = torch.min(targets).item()

        self.loss = self.criterion(predictions, targets)
        self.loss.data = self.loss.data / (max_rate - min_rate)
        
