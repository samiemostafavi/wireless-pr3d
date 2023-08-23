#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import math
import numpy as np
from loguru import logger
#from pyspark.sql import SparkSession
from pr3d.de import ConditionalRecurrentGaussianMM,ConditionalGaussianMM


enable_training = False
#parquet_file = "timeseries/r1ep5g_results/10-42-3-2_55500_20230726_171830.parquet"
parquet_file = "timeseries/r1ep5g_results/"
covariate = "netinfodata.CSQ"
target_var = "delay.send"

from pathlib import Path
import pandas as pd

data_dir = Path(parquet_file)
df = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)

"""
spark = (
    SparkSession.builder.master("local")
    .appName("LoadParquets")
    .config("spark.executor.memory", "6g")
    .config("spark.driver.memory", "70g")
    .config("spark.driver.maxResultSize", 0)
    .getOrCreate()
)

# find dataframe with the desired condition
# inputs: exp_args["condition_nums"]
df = spark.read.parquet(parquet_file)
total_count = df.count()
"""


logger.info(f"Parquet file {parquet_file} is loaded.")
logger.info(f"Total number of samples in this empirical dataset: {len(df)}")


# Check packet_multiply
# Get the first row of the DataFrame for packet multiply
#first_row = df.first()
packet_multiply = df['packet_multiply'].iloc[0]
logger.info(f"Packet multiply: {packet_multiply}")

# get all latency measurements
#df = df.na.drop("any")
#ts_latency = df.rdd.map(lambda x: x[target_var]).collect()
ts_latency = df.loc[:,target_var]
ts_latency = np.array(ts_latency)/1e6

# get all CQI measurements
#cv_csq = df.rdd.map(lambda x: x[covariate]).collect()
cv_csq = df.loc[:,covariate]
cv_csq = np.array(cv_csq)
cv_csq = cv_csq.astype(np.float64)


if packet_multiply > 1:
    # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
    ts_latency = ts_latency[packet_multiply-1::packet_multiply]
    cv_csq = cv_csq[packet_multiply-1::packet_multiply]

logger.info(f"The number of CQI samples: {len(cv_csq)}, latency samples: {len(ts_latency)}")

# load the rnn model
rmodel = ConditionalRecurrentGaussianMM(h5_addr="model_cond_rnn_gmm.h5")
#model.core_model._model.summary()
logger.success(f"RMDN GMM Model loaded successfully, number of taps: {rmodel.recurrent_taps}")

# Create inputs (X,Y) and label (y)
Y, y, X = [], [], []
for i in range(len(ts_latency) - rmodel.recurrent_taps):
    # target sequence
    Y.append(ts_latency[i:i+rmodel.recurrent_taps])
    # covariate sequence
    X.append(cv_csq[i:i+rmodel.recurrent_taps])
    # target value
    y.append(ts_latency[i+rmodel.recurrent_taps])
Y = np.array(Y)
X = np.array(X)
y = np.array(y)

logger.info(f"The number of CQI sequential samples: {len(X[:,0])}, latency sequential samples: {len(y)}")

# load the mm model
model = ConditionalGaussianMM(h5_addr="model_cond_gmm.h5")
#model.core_model._model.summary()
logger.success(f"MDN GMM Model loaded successfully.")

# evaluation starts here
# number of test cases
test_cases_num = 800000
delay_targets = [32.0,40.0,45.0] # in ms

# select a batch of sequences and a batch of targets, print the result
indxes = random.sample(range(Y.shape[0]),test_cases_num)
batchY = Y[indxes,:]
batchX = X[indxes,:]
batchy = y[indxes]

baselines = []
for delay_target in delay_targets:
    baseline = np.sum(
        np.where(batchy - delay_target*np.ones(test_cases_num,dtype=np.float64) > 0, 1, 0)
    )/np.float64(test_cases_num)
    logger.info("baseline: {:1.2e}".format(baseline))
    baselines.append(baseline)

    # calc prob rmdn
    # split data
    batch_size = 2048
    divisions = math.ceil(float(len(batchY))/float(batch_size))
    batchYm = np.array_split(batchY,divisions)
    batchXm = np.array_split(batchX,divisions)
    result = []
    for i in range(divisions):
        Yinp = np.array(batchYm[i])
        Xinp = np.array(batchXm[i])
        tmpres = rmodel.prob_batch(
                Y=Yinp,
                X=Xinp,
                y=delay_target*np.ones(len(Yinp),dtype=np.float64),
        )
        tmpres = tmpres[2]
        result = [*result,*tmpres]
    result = np.array(result)
    
    result_without_nan = result[~np.isnan(result)]
    rmdn_prob = 1.00 - np.mean(result_without_nan)
    logger.info("RMDN: {:1.2e}".format(rmdn_prob))

    # calc prob rmdn
    result = model.prob_batch(
            x=np.array([[row[-1]] for row in batchX]),
            y=delay_target*np.ones(test_cases_num,dtype=np.float64),
    )
    mdn_prob = 1.00 - np.sum(result[2])/np.float64(test_cases_num)
    logger.info("MDN: {:1.2e}".format(mdn_prob))




