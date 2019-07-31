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
