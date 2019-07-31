import json
import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from minari.evaluator import Predictor
from minari.loss import RMSELoss
from minari.dataset import SolarDataSet

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the model which you are going to test.', default='../model/test.pt')
parser.add_argument('--pid', help='Plant id that you are going to test.', type=list, default=[['3']], nargs='*')
parser.add_argument('--start_day', help='Start date that you are going to test. format is YYYY-MM-DD', default='2018-06-15')
parser.add_argument('--plot', help='Flag to decide plot the results', action='store_true')
parser.add_argument('--end_day', help='End date that you are going to test. format is YYYY-MM-DD', default=None)
args = parser.parse_args()

start_date = datetime.strptime(args.start_day, '%Y-%m-%d')
start_date = start_date.year * 10000 + start_date.month * 100 + start_date.day
end_date = args.end_day
if end_date:
    end_date = datetime.strptime(args.end_day, '%Y-%m-%d')
    end_date = end_date.year * 10000 + end_date.month * 100 + end_date.day

pid = []
pid_list = args.pid
for idx in pid_list:
    pid.append(int(idx[0]))

dataset = SolarDataSet('../data/train.csv', start_date=start_date, end_date=end_date, pid=pid)
device = torch.device('cpu')
predictor = Predictor(dataset, RMSELoss(), device)

model = torch.load(args.model, map_location='cpu')
model.eval()

result, loss = predictor.predict(model)
result = result.detach().numpy()

print('Loss: ', loss.data)

ret = {
    "name": "영암 발전소",
    "point_list": []
}

for i in range(dataset.x_data.shape[0]):
    year = int(round(dataset.x_data[:, [8]][i][0] * 2100))
    month = int(round(dataset.x_data[:, [9]][i][0] * 12))
    day = int(round(dataset.x_data[:, [10]][i][0] * 31))

    json_row = {
        "time" : int(round(dataset.x_data[:, [0]][i][0] * 24)),
        "date" : year * 10000 + month * 100 + day,
        "power" : float(result[i][0])
    }

    ret['point_list'].append(json_row)

ret = json.dumps(ret)

if args.plot:
    plt.plot(dataset.x_data[:, [0]], dataset.y_data, 'r.', label='Original Data')
    plt.plot(dataset.x_data[:, [0]], result, 'b.', label='Prediction')
    plt.legend(loc='best')
    plt.show()
