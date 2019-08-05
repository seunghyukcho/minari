import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SolarDataSet(Dataset):
    def __init__(self, path, start_date=None, end_date=None, pid=None):
        super(SolarDataSet, self).__init__()
        columns_title = ['power', 'time', 'pid', 'date', 'daylight', 'groundtemp', 'humidity',
                         'latitude', 'longitude', 'sid', 'precip', 'radiation',
                         'temperature', 'winddir', 'windspeed']
        self.start_date = start_date
        self.end_date = end_date
        self.pid = pid

        df = pd.read_csv(path)
        df = df.reindex(columns=columns_title)
        df = self.normalize(df)

        raw_data = np.array(df.values, dtype='f')

        self.len = raw_data.shape[0]
        self.x_data = raw_data[:, 1:]
        self.y_data = raw_data[:, [0]]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def normalize(self, df):
        if self.pid:
            df = df[df['pid'].isin(self.pid)]
        if self.start_date:
            if not self.end_date:
                df = df[df['date'] == self.start_date]
            else:
                df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        df['year'] = (df['date'] - df['date'] % 10000) / 10000
        df['month'] = (df['date'] % 10000 - df['date'] % 100) / 100
        df['day'] = df['date'] % 100

        ret = df.drop(columns=['date', 'sid', 'pid', 'daylight', 'precip', 'radiation', 'winddir', 'windspeed'])

        ret['year'] = ret['year'] / 2100
        ret['month'] = ret['month'] / 12
        ret['day'] = ret['day'] / 31
        ret['time'] = ret['time'] / 2400

        ret['latitude'] = (ret['latitude'] + 90) / 180
        ret['longitude'] = (ret['longitude'] + 180) / 360

        ret['humidity'] = ret['humidity'] / 100
        ret['temperature'] = (ret['temperature'] + 273) / 373

        return ret
