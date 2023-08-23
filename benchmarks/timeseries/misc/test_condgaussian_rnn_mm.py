#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random, json
import tensorflow as tf
import numpy as np
from loguru import logger
from pyspark.sql import SparkSession
from pr3d.de import ConditionalRecurrentGaussianMM
from pyspark.sql.functions import col, mean, randn

enable_training = False
parquet_file = "timeseries/r1ep5g_results/10-42-3-2_55500_20230726_171830.parquet"
target_var = "delay.send"
covariate = "netinfodata.CSQ"
normalize = [covariate]

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
logger.info(f"Parquet file {parquet_file} is loaded.")
logger.info(f"Total number of samples in this empirical dataset: {total_count}")

# preprocess add noise and standardize
noise_seed = random.randint(0,1000)
lat_mean = df.select(mean(target_var)).collect()[0][0]
lat_scale = 1e-6
noise_variance = 1
noise_highvariance = 3
df = df.withColumn(target_var+'.scaled', (col(target_var) * lat_scale))
df = df.withColumn(target_var+'.standard', ((col(target_var)-lat_mean) * lat_scale))
df = df.withColumn(target_var+'.noisy', col(target_var+'.standard') + (randn(seed=noise_seed)*noise_variance))
df = df.withColumn(target_var+'.verynoisy', col(target_var+'.standard') + (randn(seed=noise_seed)*noise_highvariance))
target_var = target_var+'.noisy'
means_dict = {
    target_var:{
        "mean":lat_mean,
        "scale":lat_scale
    }
}

if normalize:
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType
    from pyspark.sql.functions import min, max

    # UDF for converting column type from vector to double type
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

    # Iterating over columns to be scaled
    for i in normalize:
        # VectorAssembler Transformation - Converting column to vector type
        assembler = VectorAssembler(inputCols=[i],outputCol=i+".vect")

        # MinMaxScaler Transformation
        scaler = MinMaxScaler(inputCol=i+".vect", outputCol=i+".normed")

        # Pipeline of VectorAssembler and MinMaxScaler
        pipeline = Pipeline(stages=[assembler, scaler])

        # Fitting pipeline on dataframe
        df = pipeline.fit(df).transform(df).withColumn(i+".normed", unlist(i+".normed")).drop(i+".vect")
        
        imin = df.agg(min(i)).collect()[0][0]
        imax = df.agg(max(i)).collect()[0][0]
        means_dict = { i:{ "min":imin, "max":imax }, **means_dict }

# save json file
with open(
    "model_cond_rnn_gmm.json",
    "w",
    encoding="utf-8",
) as f:
    f.write(json.dumps(means_dict, indent = 4))

# Check packet_multiply
# Get the first row of the DataFrame for packet multiply
first_row = df.first()
packet_multiply = first_row["packet_multiply"]
logger.info(f"Packet multiply: {packet_multiply}")

# get all latency measurements
measurements = df.rdd.map(lambda x: x[target_var]).collect()
time_series_data = np.array(measurements)/1e6

# get all CQI measurements
covariate_csq = df.rdd.map(lambda x: x[covariate]).collect()
covariate_csq = np.array(covariate_csq)
covariate_csq = covariate_csq.astype(np.float64)

if packet_multiply > 1:
    # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
    time_series_data = time_series_data[packet_multiply-1::packet_multiply]
    covariate_csq = covariate_csq[packet_multiply-1::packet_multiply]


logger.info(f"The number of samples after removing packet duplicates: {len(time_series_data)}")

if enable_training:

    recurrent_taps = 64
    epochs = 300
    batch_size = 4096/2
    num_training_samples = 100000

    model = ConditionalRecurrentGaussianMM(
        centers=8,
        x_dim=["CSQ"],
        recurrent_taps=recurrent_taps,
        hidden_layers_config={ 
            "hidden_lstm_1": { "type": "lstm","size": 64, "return_sequences": True },
            "hidden_lstm_2": { "type": "lstm","size": 64, "return_sequences": False },
            "hidden_dense_1": { "type": "dense","size": 32,"activation": "tanh" },
            "hidden_dense_2": { "type": "dense","size": 32,"activation": "tanh" }, 
        }
    )

    # limit the number of samples
    time_series_data = time_series_data[:num_training_samples]
    covariate_csq = covariate_csq[:num_training_samples]
    logger.info(f"Limited the number of samples for training: {len(time_series_data)}")

    # number of taps
    num_taps = recurrent_taps

    # Create input (Y) and target (y) data
    Y, y, X = [], [], []
    for i in range(len(time_series_data) - num_taps):
        # target sequence
        Y.append(time_series_data[i:i+num_taps])
        # covariate sequence
        X.append(covariate_csq[i:i+num_taps])
        # target value
        y.append(time_series_data[i+num_taps])
    Y = np.array(Y)
    X = np.array(X)
    y = np.array(y)

    # Reshape the input data for LSTM (samples, time steps, features)
    Y = Y.reshape(Y.shape[0], num_taps, 1)
    X = X.reshape(X.shape[0], num_taps, 1)

    # Split the data into training and testing sets (adjust the split ratio as needed)
    split_ratio = 0.8
    split_index = int(len(Y) * split_ratio)
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    logger.info(f"Number of training sequences: {len(y_train)}")
    logger.info(f"Number of test sequences: {len(y_test)}")

    model.training_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            #learning_rate=learning_rate,
        ),
        loss=model.loss,
    )

    # find steps_per_epoch
    steps_per_epoch = len(y_train) // batch_size
    # train!
    model.training_model.fit(
        x=[Y_train, X_train, y_train],
        y=y_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
    )

    # Evaluate the model
    loss = model.training_model.evaluate(
        x=[Y_test, X_test, y_test], 
        y=y_test
    )
    print(f"Test Loss: {loss}")

    # Save the model to a file
    # training done, save the model
    model.save("model_cond_rnn_gmm.h5")
    logger.success("Model saved successfully.")


model = ConditionalRecurrentGaussianMM(h5_addr="model_cond_rnn_gmm.h5")
model.core_model._model.summary()
logger.success("Model loaded successfully.")

# number of taps
num_taps = model.recurrent_taps
print(num_taps)

# Create input (Y) and target (y) data
Y, y, X = [], [], []
for i in range(len(time_series_data) - num_taps):
    # target sequence
    Y.append(time_series_data[i:i+num_taps])
    # covariate sequence
    X.append(covariate_csq[i:i+num_taps])
    # target value
    y.append(time_series_data[i+num_taps])
Y = np.array(Y)
X = np.array(X)
y = np.array(y)


# select a single sequence and check probability
indx = random.choice(range(Y.shape[0]))
singleY = Y[indx,:]
singleX = X[indx,:]
singley = 10 #ms
logger.info(f"check the probability of X:{singleX}, Y:{singleY} at {singley} ms")
result = model.prob_single(singleY,singleX,singley)
logger.success(f"pdf:{result[0]}, log_pdf:{result[1]}, ecdf:{result[2]}")


# use the previous sequences and sample the resulting distribution 20 times in parallel
logger.info(f"produce 20 parallel samples from X:{singleX}, Y:{singleY}")
result = model.sample_n_parallel(singleY,singleX,20)
logger.success(f"parallel samples: {result}")

# select a batch of sequences and a batch of targets, print the result
batch_size = 8
indxes = random.choices(range(Y.shape[0]),k=batch_size)
batchY = Y[indxes,:]
batchX = X[indxes,:]
batchy = np.array([10,12,14,16,18,20,22,24])
logger.info(f"check the probabilities of a batch of size {batch_size}, X:{batchX}, Y:{batchY} at {batchy} ms")
result = model.prob_batch(batchY,batchX,batchy,batch_size=batch_size)
logger.success(f"pdf:{result[0]}, log_pdf:{result[1]}, ecdf:{result[2]}")



