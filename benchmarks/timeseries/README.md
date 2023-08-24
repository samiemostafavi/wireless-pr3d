# README

The scripts work by reading the configuration from a json file. Therefore, `CONF_FILE_ADDR` is a required env variable that must be set for running the scripts as below:
```
CONF_FILE_ADDR=timeseries/trainconf_cond_rec_gmm.json python timeseries/train.py
CONF_FILE_ADDR=timeseries/evalconf.json python timeseries/evaluate.py
```
You can use `CPU_ONLY=true` as well to force the code ignore GPU.


# Traning durations

14,059 parameters LSTM network took 4.5 hours to train. 