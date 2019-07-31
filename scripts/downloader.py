import argparse
import requests
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='directory path where you are going to save the dataset', default='../data')
args = parser.parse_args()

base_url = 'http://portal.triple3e.me:30000/solar_pred?query={plant' \
           '{pid sid date time winddir windspeed daylight power humidity precip radiation' \
           ' temperature groundtemp latitude longitude}}'

print('Downloading... It will take a few minutes')
raw_data = requests.get(base_url)

data = raw_data.json()
df = pd.DataFrame(data['data']['plant'])

if not os.path.exists(args.dir):
    os.mkdir(args.dir)

df.to_csv(args.dir + '/train.csv', mode='w', index=False)
print('Finish!')
