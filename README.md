# ECG Multi-Label Classfication

This repo is to archive code of a competition I attented.  
The competition is about multi-label classfication of 8-leads ECGs.  
This repo does not provide a top solution, but for backup and sharing.


## 1. Dataset

Page: ["合肥高新杯"心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction?spm=5176.12281915.0.0.35216dc9Wdmtzg)

|Round|Classes|Train\*|TestA\*|TestB\*|
|:---:|:-----:|:---:|:---:|:---:|
|Round 1|55|24106|8036|5435|
|Round 2|34|20036|9918|12622|

\* Number of samples


```Round 2 TestA``` and ```Round 2 TestB``` are not available for downloading.  
```Round 1 TestB``` does not have labels.  
Access to ```Round 1 Train```, ```Round 1 TestA``` and ```Round2 Train``` should be required.


## 2. Code and Command

|Round|Path|Classes|TestA\*|TestB\*|Rank|
|:---:|:--:|:-----:|:-----:|:-----:|:--:|
|Round 1|[[Link]](./round1)|55|0.8201|0.8424|28/2353|
|Round 2|[[Link]](./round2)|34|0.9251|0.9225|35/2353|

\* F1-score


## 3. Brief Intro

- **Preprocessing**:
  - merge ```Round 1 Train``` and ```Round 1 TestA```, remove duplicates;
  - compute additional 4 leads (III, aVR, aVL, aVF) on basis of given 8 leads;
  - standard normalization;
  - (only in round 2) compute weights for samples in ```Round 1 data```, merged with ```Round 2 Train```.
- **Model Structure**:
  - 1D DenseNet with slight modification.
- **Train**:
  - 5-fold split by scikit-multilearn;
  - 5-fold cross validation;
  - augmentation: slightly adjust amplitude and baseline of ECG;
  - compound loss function: F1Loss, FocalLoss and MultiLabelSoftMarginLoss;
  - search best threshold on validation set after each fold training.
- **Prediction**:
  - ensemble 5-fold predictions by votting.


## 4. Dependencies

|Package|Version|Comment|
|:-----:|:-----:|:-----:|
|conda|4.5.11||
|python|3.7.3||
|tqdm|4.32.1||
|numpy|1.16.4|conda install|
|scipy|1.3.0||
|pandas|0.24.2||
|pytorch|1.1.0|conda install with cuda 9.0|
|scikit-learn|0.21.2||
|scikit-multilearn|0.2.0|pip install scikit-multilearn|
