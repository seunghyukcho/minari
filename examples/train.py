import logging
import argparse
from minari.trainer import Trainer
from minari.dataset import SolarDataSet
from minari.loss import MAELoss
from minari.models import SimpleNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path which you are going to save the model.', default='../model')
parser.add_argument('--model', help='Model name you are going to save.', default='test')
parser.add_argument('--pid', help='Plant id that you are going to test.', type=list, default=[['8']], nargs='*')
parser.add_argument('--lr', help='Learning rate', default='0.01')
parser.add_argument('--epoch', help='Epoch size', default='10')
parser.add_argument('--batch', help='Batch size', default='64')
parser.add_argument('--dataset', help='Path to the dataset file going to use in training.', default='../data/train.csv')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
output_path = args.path
model_name = args.model
lr = float(args.lr)
epoch_size = int(args.epoch)
batch_size = int(args.batch)

pid = []
pid_list = args.pid
for idx in pid_list:
    pid.append(int(idx[0]))

dataset = SolarDataSet(args.dataset, pid=pid)
model = SimpleNetwork(dataset.x_data.shape[1])

trainer = Trainer(loss=MAELoss())
trainer.train(dataset=dataset, model=model, output_dir=output_path,
              file_name=model_name, epoch_size=epoch_size, learning_rate=lr)
