import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from minari.evaluator import Predictor, TwoLevelPredictor
from minari.loss import MAELoss
from minari.dataset import SolarDataSet, TwoLevelDataSet


def parse_datestring(datestr):
    if datestr:
        datestr = datetime.strptime(datestr, '%Y-%m-%d')
        datestr = datestr.year * 10000 + datestr.month * 100 + datestr.day

    return datestr


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the model which you are going to test.', default='../model/test.pt')
parser.add_argument('--pid', help='Plant id that you are going to test.', type=list, default=[['8']], nargs='*')
parser.add_argument('--start_date', help='Start date that you are going to test. format is YYYY-MM-DD', default='2018-06-15')
parser.add_argument('--plot', help='Flag to decide plot the results', action='store_true')
parser.add_argument('--end_date', help='End date that you are going to test. format is YYYY-MM-DD', default=None)
args = parser.parse_args()

start_date = parse_datestring(args.start_date)
end_date = parse_datestring(args.end_date)

pid = []
pid_list = args.pid
for idx in pid_list:
    pid.append(int(idx[0]))

dataset = TwoLevelDataSet('../data/train.csv', start_date=start_date, end_date=end_date, pid=pid)
device = torch.device('cpu')
predictor = TwoLevelPredictor(dataset, MAELoss(), device)

model = torch.load(args.model, map_location='cpu')
model.eval()

result, loss = predictor.predict(model)
result = result.detach().numpy()

print('Loss: ', loss.data)

if args.plot:
    plt.plot(dataset.x1_data[:, [0]], dataset.y_data, 'r.', label='Original Data')
    plt.plot(dataset.x1_data[:, [0]], result, 'b.', label='Prediction')
    plt.legend(loc='best')
    plt.show()
