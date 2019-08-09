from minari.trainer import Trainer
from torch.autograd import Variable


class TwoLevelTrainer(Trainer):
    def _train_batch(self, model, data, device):
        x1_data, x2_data, y_data = data
        x1_data, x2_data, y_data = Variable(x1_data.to(device)), \
                                      Variable(x2_data.to(device)), Variable(y_data.to(device))

        self.optimizer.zero_grad()
        results = model(x1_data, x2_data)

        loss = self.loss.compute_loss(results, y_data)
        self.loss.backward()
        self.optimizer.step()

        return loss.data
