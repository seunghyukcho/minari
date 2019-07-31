import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from minari.evaluator import ChargeCalculator
from minari.loss import RMSELoss
from minari.dataset import SolarDataSet

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the model which you are going to test.', default='../model/test.pt')
parser.add_argument('--pid', help='Plant ids that you are going to test.', default=[['3']], type=list, nargs='*')
parser.add_argument('--start_date', help='Start date that you are going to test. format is YYYY-MM-DD', default='2018-06-15')
parser.add_argument('--end_date', help='End date that you are going to test. format is YYYY-MM-DD', default=None)
parser.add_argument('--plot', help='Flag to decide plot the results', action='store_true')
args = parser.parse_args()

start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
start_date = start_date.year * 10000 + start_date.month * 100 + start_date.day

end_date = args.end_date
if end_date:
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    end_date = end_date.year * 10000 + end_date.month * 100 + end_date.day

pid = []
pid_list = args.pid
for idx in pid_list:
    pid.append(int(idx[0]))

dataset = SolarDataSet('../data/train.csv', start_date=start_date, end_date=end_date, pid=pid)
device = torch.device('cpu')
calculator = ChargeCalculator(dataset, device)

model = torch.load(args.model, map_location='cpu')
model.eval()

result = calculator.calculate(model)
msg = 'Electricity charge from %s' % args.start_date
if args.end_date:
    msg += ' to %s' % args.end_date
msg += ': ' + str(result) + ' won'

print(msg)
