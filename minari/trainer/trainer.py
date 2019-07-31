import logging
import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from minari.optimizer import AdamOptimizer


class Trainer(object):
    def __init__(self, loss, batch_size=64):
        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = None

        self.logger = logging.getLogger('Trainer')

    def _train_batch(self, model, data, device):
        x_datas, y_datas = data
        x_datas, y_datas = Variable(x_datas.to(device)), Variable(y_datas.to(device))

        self.optimizer.zero_grad()
        results = model(x_datas)

        loss = self.loss.compute_loss(results, y_datas)
        self.loss.backward()
        self.optimizer.step()

        return loss.data

    def _train_epoch(self, model, train_loader, total_steps, device):
        total_loss = 0

        progress_bar = tqdm(train_loader)
        for step, data in enumerate(progress_bar):
            batch_loss = self._train_batch(model, data, device)
            total_loss += batch_loss
            progress_bar.set_description('Batch loss %.4f' % batch_loss)

        return total_loss

    def train(self, dataset, model, output_dir, file_name, epoch_size=10, optimizer=None, learning_rate=0.01):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        total_rows = dataset.x_data.shape[0]
        total_steps = (total_rows - 1) / self.batch_size + 1

        model = model.to(device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamOptimizer(model, learning_rate=learning_rate)

        for i in range(epoch_size):
            epoch_loss = self._train_epoch(model, train_loader, total_steps, device)

            log_msg = '[Epoch %d] finished. Loss: %.4f\n' % (i + 1, epoch_loss / total_steps)
            self.logger.info(log_msg)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        torch.save(model, output_dir + '/' + file_name + '.pt')


