# Solar Prediction
[![pipeline status](https://git.triple3e.com/triple3e-dev/solar-prediction/badges/master/pipeline.svg)](https://git.triple3e.com/triple3e-dev/solar-prediction/commits/master)
[![coverage report](https://git.triple3e.com/triple3e-dev/solar-prediction/badges/master/coverage.svg)](https://git.triple3e.com/triple3e-dev/solar-prediction/commits/master)

**태양광 발전량 예측**을 위한 repository 입니다. 기본 알고리즘은 딥러닝으로, 다양한 neural network를 실험하고 있습니다. 현재 [미국의 발전량 데이터](https://www.nrel.gov/grid/solar-power-data.html)로 학습을 하고 있습니다. 어느 정도 정확도를 보이면, 한국 데이터에 적용할 계획입니다.

## Getting Started

### Prerequisites

아래 내용들을 알고 계셔야 프로젝트를 이해하실 수 있습니다.

- [딥러닝의 기본]([http://hunkim.github.io/ml/](http://hunkim.github.io/ml/)) 정도의 지식을 가지고 있다.
- [Pytorch](https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/beginner/deep_learning_60min_blitz.html)를 사용하여 neural network를 만들고, test 할 수 있다.
- pipenv, 또는 pycharm을 사용하여 **가상환경**을 만들 수 있다.

위 내용을 자세히 모르신다면, 링크를 통해 배우실 수 있습니다.

### Installing

다음 조건을 만족하는 가상 환경, 또는 global python을 사용하시면 됩니다.

- python3.7
- pandas, torch, matplotlib.pyplot, numpy module 설치

이후에는 python command를 이용하여 다음 프로그램들을 실행할 수 있습니다. 다만 train하는데 사용되는 파일들이 업로드 되지 않아, 실제로 실행이 안될 것입니다. 온라인으로 받을 수 있는 방법을 준비 중입니다.

- python train.py : 실제 태양광 데이터를 사용하여 모델을 학습할 수 있습니다. 모델은 model.py에 있습니다.
- python test.py : 하루 동안의 데이터를 모델로 예측한 결과를 그래프로 확인하실 수 있습니다.

![Imgur](https://i.imgur.com/rQ74ngp.png)

## Report

실제 모델들의 코드는 tag를 통해 확인하실 수 있습니다. 표에 드러나지 않는 정보는 다음과 같습니다.


| 이름                  | 비고 |
| --------------------- | ---------------- | 
| simple-neural-network-relu | [링크](https://git.triple3e.com/triple3e-dev/solar-prediction/issues/3)   |
| simple-neural-network-leaky-relu | [링크](https://git.triple3e.com/triple3e-dev/solar-prediction/issues/4) |

## Authors

* **조승혁** (포스텍 컴퓨터공학과)

## TODO

- command line argument들에 대한 설명 추가
- 학습 종료 조건 개선
