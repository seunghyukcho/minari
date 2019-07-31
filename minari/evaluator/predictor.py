from torch.autograd import Variable
import torch


class Predictor(object):
    def __init__(self, dataset, loss, device):
        self.dataset = dataset
        self.loss = loss
        self.device = device

    def predict(self, model):
        x_datas, y_datas = self.dataset.x_data, self.dataset.y_data
        x_datas, y_datas = Variable(torch.tensor(x_datas).to(self.device)), Variable(torch.tensor(y_datas).to(self.device))

        outputs = model(x_datas)
        result = self.loss.compute_loss(outputs, y_datas)

        return outputs, result
