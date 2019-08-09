import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from minari.dataset import SolarDataSet, TwoLevelDataSet
from minari.evaluator import Predictor, TwoLevelPredictor
from minari.loss import MAELoss


def parse_datestring(datestr):
    if datestr:
        datestr = datetime.strptime(datestr, '%Y-%m-%d')
        datestr = datestr.year * 10000 + datestr.month * 100 + datestr.day

    return datestr


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the model which you are going to test.', default='../model/test.pt')
parser.add_argument('--pid', help='Plant id that you are going to test.', type=list, default=[['8']], nargs='*')
parser.add_argument('--start_date', help='Start date that you are going to test. format is YYYY-MM-DD', default=None)
parser.add_argument('--plot', help='Flag to decide plot the results', action='store_true')
parser.add_argument('--mode', help='Select your model mode. 1: one level, 2: two level', default=1)
parser.add_argument('--end_date', help='End date that you are going to test. format is YYYY-MM-DD', default=None)
args = parser.parse_args()

start_date = parse_datestring(args.start_date)
end_date = parse_datestring(args.end_date)
mode = int(args.mode)
pid_list = args.pid

pid = []
for idx in pid_list:
    pid.append(int(idx[0]))

device = torch.device('cpu')

dataset = SolarDataSet('../data/train.csv', start_date=start_date, end_date=end_date, pid=pid)
predictor = Predictor(dataset, MAELoss(), device)

if mode is 2:
    dataset = TwoLevelDataSet('../data/train.csv', start_date=start_date, end_date=end_date, pid=pid)
    predictor = TwoLevelPredictor(dataset, MAELoss(), device)

model = torch.load(args.model, map_location='cpu')
model.eval()

result, loss = predictor.predict(model)
result = result.detach().numpy()

print('Loss: ', loss.data)

if args.plot:
    plt.plot(dataset.x_axis, dataset.y_data, 'r.', label='Original Data')
    plt.plot(dataset.x_axis, result, 'b.', label='Prediction')
    plt.legend(loc='best')
    plt.show()
