from torch.autograd import Variable
import torch


class Evaluator(object):
    def __init__(self, dataset, criterion, device):
        self.dataset = dataset
        self.criterion = criterion
        self.device = device

    def evaluate(self, model):
        x_datas, y_datas = self.dataset.x_data, self.dataset.y_data
        x_datas, y_datas = Variable(torch.tensor(x_datas).to(self.device)), Variable(torch.tensor(y_datas).to(self.device))

        outputs = model(x_datas)
        loss = torch.sqrt(self.criterion(outputs, y_datas))

        return loss.data




