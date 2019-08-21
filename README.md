# Minari

Minari is a python package that helps you to achieve high accuracy in solar prediction.

## Installation
I've only tested the package in python3, so python2 is not recommended. Also, we recommend creating a new virtual environment for this project (using virtualenv or conda).

### Install from source
I prepared `requirements.txt` for dependencies. It will make you easier to prepare the environment for this project.
```bash
$ pip install -r requirements.txt
$ python setup.py install
```

### Getting Started
#### Prepare solar pv dataset
```bash
# The generated data is stored in data/data.csv by default
$ scripts/download.sh
```
Now, we have 8 different solar pv dataset from different plants. They are distinguished by a unique number, `pid`. Nowadays, the measurement is different for each other, so it is not recommended to use a mixture of different number of pid.

#### Train and see the results
```bash
$ cd examples
$ python train.py -h # Model will be saved in ../model/test.pt as default.
usage: train.py [-h] [--path PATH] [--model MODEL] [--pid [PID [PID ...]]]
                [--lr LR] [--epoch EPOCH] [--batch BATCH]
                [--start_date START_DATE] [--end_date END_DATE]
                [--dataset DATASET] [--mode MODE]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path which you are going to save the model.
  --model MODEL         Model name you are going to save.
  --pid [PID [PID ...]]
                        Plant id that you are going to test.
  --lr LR               Learning rate
  --epoch EPOCH         Epoch size
  --batch BATCH         Batch size
  --start_date START_DATE
                        Start date that you are going to train. format is
                        YYYY-MM-DD
  --end_date END_DATE   End date that you are going to train. format is YYYY-
                        MM-DD
  --dataset DATASET     Path to the dataset file going to use in training.
  --mode MODE           Select your model mode. 1: one level, 2: two level
```
To see the prediction results, use `predict.py`.
```bash
$ python predict.py -h
usage: predict.py [-h] [--model MODEL] [--pid [PID [PID ...]]]
                  [--start_date START_DATE] [--plot] [--mode MODE]
                  [--end_date END_DATE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to the model which you are going to test.
  --pid [PID [PID ...]]
                        Plant id that you are going to test.
  --start_date START_DATE
                        Start date that you are going to test. format is YYYY-
                        MM-DD
  --plot                Flag to decide plot the results
  --mode MODE           Select your model mode. 1: one level, 2: two level
  --end_date END_DATE   End date that you are going to test. format is YYYY-
                        MM-DD
```
Using plot option, you can see a graph of your prediction.
