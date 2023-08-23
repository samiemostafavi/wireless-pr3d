#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random, json
import tensorflow as tf
import numpy as np
from loguru import logger
from pyspark.sql import SparkSession
from pr3d.de import ConditionalGaussianMM
from pyspark.sql.functions import col, mean, randn, min, max, udf
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

enable_training = False
parquet_file = "timeseries/r1ep5g_results/10-42-3-2_55500_20230726_171830.parquet"
target_var = "delay_send"
normalize = ["netinfodata_CSQ"]
covariate = "netinfodata_CSQ_normed"
filename = "model_cond_rgmm"

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

#change column name, replace . with _
for mcol in df.columns:
    df = df.withColumnRenamed(mcol, mcol.replace(".", "_"))

# preprocess add noise and standardize
noise_seed = random.randint(0,1000)
lat_mean = df.select(mean(target_var)).collect()[0][0]
lat_scale = 1e-6
noise_variance = 1
noise_highvariance = 3
df = df.withColumn(target_var+'_scaled', (col(target_var) * lat_scale))
df = df.withColumn(target_var+'_standard', ((col(target_var)-lat_mean) * lat_scale))
df = df.withColumn(target_var+'_noisy', col(target_var+'_standard') + (randn(seed=noise_seed)*noise_variance))
df = df.withColumn(target_var+'_verynoisy', col(target_var+'_standard') + (randn(seed=noise_seed)*noise_highvariance))
target_var = target_var+'_noisy'
means_dict = {
    target_var:{
        "mean":lat_mean,
        "scale":lat_scale
    }
}

if normalize:
    # UDF for converting column type from vector to double type
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    # Iterating over columns to be scaled
    for i in normalize:
        # convert covariate column to float
        df = df.withColumn(i, df[i].cast(DoubleType()))
        # VectorAssembler Transformation - Converting column to vector type
        assembler = VectorAssembler(inputCols=[i],outputCol=i+"_vect")
        # MinMaxScaler Transformation
        scaler = MinMaxScaler(inputCol=i+"_vect", outputCol=i+"_normed")
        # Pipeline of VectorAssembler and MinMaxScaler
        pipeline = Pipeline(stages=[assembler, scaler])
        # Fitting pipeline on dataframe
        df = pipeline.fit(df).transform(df).withColumn(i+"_normed", unlist(i+"_normed")).drop(i+"_vect")
        imin = df.agg(min(i)).collect()[0][0]
        imax = df.agg(max(i)).collect()[0][0]
        means_dict = { i:{ "min":imin, "max":imax }, **means_dict }
    
# save json file
with open(
    filename + ".json",
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
time_series_data = np.array(measurements)

# get all CQI measurements
covariate_csq = df.rdd.map(lambda x: x[covariate]).collect()
covariate_csq = np.array(covariate_csq)

if packet_multiply > 1:
    # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
    time_series_data = time_series_data[packet_multiply-1::packet_multiply]
    covariate_csq = covariate_csq[packet_multiply-1::packet_multiply]


logger.info(f"The number of samples after removing packet duplicates: {len(time_series_data)}")

if enable_training:
    epochs = 50
    batch_size = 4096
    num_training_samples = 50000

    model = ConditionalGaussianMM(
        centers=8,
        x_dim=["CSQ"],
        hidden_sizes=(32,32,32),
    )

    # limit the number of samples
    time_series_data = time_series_data[:num_training_samples]
    covariate_csq = covariate_csq[:num_training_samples]
    logger.info(f"Limited the number of samples for training: {len(time_series_data)}")

    # Create X and Y
    Y = np.array(time_series_data)
    X = np.array(covariate_csq)

    # Split the data into training and testing sets (adjust the split ratio as needed)
    split_ratio = 0.8
    split_index = int(len(Y) * split_ratio)
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    X_train, X_test = X[:split_index], X[split_index:]

    logger.info(f"Number of training sequences: {len(Y_train)}")
    logger.info(f"Number of test sequences: {len(Y_test)}")

    model.training_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            #learning_rate=learning_rate,
        ),
        loss=model.loss,
    )

    # find steps_per_epoch
    steps_per_epoch = len(Y_train) // batch_size
    # train!
    model.training_model.fit(
        x=(X_train,Y_train),
        y=Y_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
    )

    # Evaluate the model
    loss = model.training_model.evaluate(
        x=[X_test, Y_test], 
        y=Y_test
    )
    print(f"Test Loss: {loss}")

    # Save the model to a file
    # training done, save the model
    model.save("model_cond_gmm.h5")
    logger.success("Model saved successfully.")


model = ConditionalGaussianMM(h5_addr="model_cond_gmm.h5")
model.core_model._model.summary()
logger.success("Model loaded successfully.")





