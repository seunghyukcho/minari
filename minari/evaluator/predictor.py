import torch
from torch.autograd import Variable


class Predictor(object):
    def __init__(self, dataset, loss, device):
        self.dataset = dataset
        self.loss = loss
        self.device = device

    def predict(self, model):
        x_data, y_data = self.dataset.x_data, self.dataset.y_data
        x_data, y_data = Variable(torch.tensor(x_data).to(self.device)), Variable(torch.tensor(y_data).to(self.device))

        outputs = model(x_data)
        result = self.loss.compute_loss(outputs, y_data)

        return outputs, result


class TwoLevelPredictor(Predictor):
    def predict(self, model):
        x1_data, x2_data, y_data = self.dataset.x1_data, self.dataset.x2_data, self.dataset.y_data
        x1_data, x2_data, y_data = Variable(torch.tensor(x1_data).to(self.device)), \
                                   Variable(torch.tensor(x2_data).to(self.device)), Variable(
            torch.tensor(y_data).to(self.device))

        outputs = model(x1_data, x2_data)
        result = self.loss.compute_loss(outputs, y_data)

        return outputs, result
