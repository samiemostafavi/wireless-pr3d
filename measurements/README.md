# Measurement Schemes

Our measurements were primarily conducted to show the latency probability distribution conditioned on:

1. Different wireless technologies 
    * Private Commercial off the shelv (COTS) 5G
    * Openairinterface software-defined radio 5G
    * Mango communications software-defined radio IEEE 802.11g
2. Different locations resulting in different SINR, RSRP, RSRQ, etc 
3. Different transmission parameters such as MCS index
4. Different link utilizations: high, mid, low untilization 
    * Packet size
    * Packets arrivals periodicity

The purpose is to find the relation between the tail behaviour and all these conditions.


## First Measurement Campaign

These measurements were presented in the first conference paper.

1. [COTS5G](./campaign1/COTS5G.md)
2. [SDR5G](./campaign1/SDR5G.md)
3. [IEEE802.11g](./campaign1/IEEE80211g.md)

## Second Measurement Campaign

We are planning to use ExPECA testbed for these measurements. More conditions will be investigated and more samples are planned to collect.


# Access/Store Measurements

All the measurements are published on Kaggle: [wireless-pr3d](https://www.kaggle.com/datasets/samiemostafavi/wireless-pr3d).

You can download them if you install Kaggle Python package and add your token
```
pip install kaggle
vim /home/wlab/.kaggle/kaggle.json
```

Then download the dataset by running
```
cd benchmarks
kaggle datasets download -d samiemostafavi/wireless-pr3d
unzip wireless-pr3d.zip
```


