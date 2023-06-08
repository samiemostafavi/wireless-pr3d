# Data-Driven Latency Probability Prediction for Wireless Networks: Focusing on Tail Probabilities

In this work, we use mixture density networks, to predict the latency of 5G wireless links, particularly for extreme latencies that impact time-critical applications. We analyze Gaussian mixture models and a novel approach that integrates extreme value models into the mixture of parametric distributions. Through our investigation, we examine the impact of the number of training samples, the complexity of the tail profile, and the generalization capabilities of these approaches. Our results demonstrate that both approaches achieve acceptable accuracy with sufficient training samples. Additionally, we find that noise regularization improves the accuracy of the fit, particularly in the case of GMEVM when the tail profile is non-smooth. 

This repository contains the instructions and scripts on:
1. How we measure end-to-end latency on the 5G network (`measurements` folder)
2. How we train and benchmark the latency prediction systems (`benchmarks` folder)
 
For the lateny prediction task, an upstream project [pr3d](https://github.com/samiemostafavi/pr3d) is used. To reproduce the paper results, you need to download the datasets and use them for training or evaluation of the latency predictors which are implemented in [pr3d](https://github.com/samiemostafavi/pr3d). [Pr3d](https://github.com/samiemostafavi/pr3d) uses Python, Tensorflow, and Keras.

The measured latencies datasets are stored on Kaggle: [wireless-pr3d version 2](https://www.kaggle.com/datasets/samiemostafavi/wireless-pr3d/versions/2).

## Goal of the work

Study the effectiveness of mixture density networks (MDN)s specifically in predicting the tail behaviour for latency prediction in wireless networks

## Aproach

1. Measurements (`measurements` folder)
2. Training predictors (`benchmarks` folder)
3. Evaluation (`benchmarks` folder)

## Methodology

Run measurements on differenct wireless networks:
- Commercial Private 5G network by Ericsson
- Software-defined radio 5G network by Openairinterface

We considered MCS index in SDR 5G as a condition to change the wireless link's latency distribution.

## Paper
This repository contains the models, evaluation schemes, and numerics of the following paper: ***Data-Driven Latency Probability Prediction for Wireless Networks: Focusing on Tail Probabilities*** published by ... [here](https://ieeexplore.ieee.org/document/?).


## Citing
If you use the results of this work in your research, please cite the following papers:
```
@INPROCEEDINGS{
}

@INPROCEEDINGS{9708928,
  author={Mostafavi, Seyed Samie and Dán, György and Gross, James},
  booktitle={2021 IEEE/ACM Symposium on Edge Computing (SEC)}, 
  title={Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models}, 
  year={2021},
  volume={},
  number={},
  pages={416-422},
  doi={10.1145/3453142.3493506}
}

```

