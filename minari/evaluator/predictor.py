from torch.autograd import Variable
import torch


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
