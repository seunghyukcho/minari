from minari.evaluator import Predictor
from torch.autograd import Variable
import torch


class TwoLevelPredictor(Predictor):
    def predict(self, model):
        x1_data, x2_data, y_data = self.dataset.x1_data, self.dataset.x2_data, self.dataset.y_data
        x1_data, x2_data, y_data = Variable(torch.tensor(x1_data).to(self.device)), \
                                   Variable(torch.tensor(x2_data).to(self.device)), Variable(torch.tensor(y_data).to(self.device))

        outputs = model(x1_data, x2_data)
        result = self.loss.compute_loss(outputs, y_data)

        return outputs, result
