## Peking University/Baidu - Autonomous Driving Competition
Simple solution of Baidu - Autonomous Driving competition
#### Model description
  The main idea of this competition is to build the model which can predict 6DoF for every car (not every vechicle!)<br/>
  My solution (69th private leaderboard) is basing on [CenterNet](https://arxiv.org/pdf/1904.07850.pdf) paper <br/>
Also I took many useful functions for image pre/postprocessing from public [kernel](https://www.kaggle.com/hocop1/centernet-baseline) (lets upvote!)

#### Instalation
```
$ pip3 install -r requirements.txt
```

#### Prepare dataset
Download and extract data from competition to main directory.<br/>
After that you need to create resized dataset - to speed up your next steps.

```
$ mkdir resized_labels
$ mkdir resized_train
$ mkdir resized_test
$ python3 create_resized_dataset.py
```
Also I recommend create folder for models and inferences (for automatizating all another steps):
```
$ mkdir models
$ mkdir inferences
```

#### Train Model

```
$ python3 train.py --encoder=b2 --fold=0 --batch_size=2 --seed=123 --max_epoch=40
```

#### Inference
```
$ python3 train.py --encoder=b2 --batch_size=2 --model_checkpoint=models/b2_fold0.pth
or
$ python3 train.py --encoder=b2 --fold=0 --batch_size=2
```
#### Visualization
![image.png](https://github.com/AYaroshevskii/Baidu-Autonomous-Driving-competition/blob/master/image.png)

#### How it can be improved
Add stronger augmentation (for example flips, but remember that 6DoF will change), play with image sizes, different losses, deeper models
