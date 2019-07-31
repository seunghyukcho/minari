import torch
import numpy
from torch.autograd import Variable


class ChargeCalculator(object):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    def calculate(self, model):
        x_datas, y_datas = self.dataset.x_data, self.dataset.y_data
        x_datas, y_datas = Variable(torch.tensor(x_datas).to(self.device)), Variable(torch.tensor(y_datas).to(self.device))
        
        outputs = model(x_datas)
        result = outputs.detach().numpy()

        ret = 0.0
        for row in result:
            power = row[0]
            if power <= 200:
                ret += power * 93.3
            elif power <= 400:
                ret += power * 187.9
            else:
                ret += power * 280.6

        return ret
