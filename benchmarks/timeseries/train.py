import os
from os import environ
from loguru import logger

CPU_ONLY = False
if environ.get('CPU_ONLY') is not None:
    val = environ.get('CPU_ONLY')
    if val.lower()=='true':
        logger.info("CPU_ONLY is set")
        CPU_ONLY = True

if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random, json
from pathlib import Path
import tensorflow as tf
import numpy as np
from pyspark.sql import SparkSession
from pr3d.de import ConditionalRecurrentGaussianMixtureEVM, ConditionalRecurrentGaussianMM, ConditionalGaussianMixtureEVM, ConditionalGaussianMM, RecurrentGaussianMM, GaussianMixtureEVM, GaussianMM, RecurrentGaussianMEVM
from pyspark.sql.functions import col, mean, randn, min, max, udf, isnan, isnull
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from datetime import datetime

def get_yymmddhhmmss():
    now = datetime.now()
    year = str(now.year)[-2:]  # Taking the last two digits of the year
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    seconds = str(now.second).zfill(2)
    milliseconds = str(now.microsecond // 1000).zfill(3)  # Milliseconds part
    microseconds = str(now.microsecond % 1000).zfill(3)  # Microseconds part
    
    return f"{year}{month}{day}{hour}{minute}{seconds}_{milliseconds}{microseconds}"


# opening training configuration
if environ.get('CONF_FILE_ADDR') is not None:
    with open(environ.get('CONF_FILE_ADDR')) as json_file:
        conf = json.load(json_file)
else:
    raise Exception("CONF_FILE_ADDR environment variable is not set.")

# preprocessing data config
parquet_files = conf["data"]["parquet_files"]
normalize = conf["data"]["normalize"]
standardize = conf["data"]["standardize"]
target_scale = conf["data"]["target_scale"]
noise_variance = conf["data"]["noise_variance"]

# creating the model config
modelname = conf["model"]["modelname"]
recurrent_taps = conf["model"]["recurrent_taps"]
target_var = conf["model"]["target_var"]
centers = conf["model"]["centers"]
conditions = conf["model"]["conditions"]
training_rounds = conf["model"]["training_rounds"]
batch_size = conf["model"]["batch_size"]
num_training_samples = conf["model"]["num_training_samples"]
model_config = conf["model"]["model_config"]

#output directory
output_dir = conf["data"]["output_dir"]
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# post config variables
recurrent = (True if recurrent_taps > 0 else False)
conditional = (True if len(conditions) > 0 else False)
modelid_str = get_yymmddhhmmss()
filename = "model_" + modelid_str

# start Spark session
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
df = spark.read.parquet(parquet_files)
total_count = df.count()
logger.info(f"Parquet files {parquet_files} are loaded.")
logger.info(f"Total number of samples in this empirical dataset: {total_count}")

#change column name, replace . with _
for mcol in df.columns:
    df = df.withColumnRenamed(mcol, mcol.replace(".", "_"))

# remove non latency values
df = df.filter(
    (col(target_var).cast("double").isNotNull()) &
    ~isnan(target_var) &
    ~isnull(target_var)
)

# preprocess add noise and standardize
means_dict = { "standardized":{} }
if standardize:
    logger.info(f"Standardizing {target_var} column")
    noise_seed = random.randint(0,1000)
    target_mean = df.select(mean(target_var)).collect()[0][0]
    df = df.withColumn(target_var+'_scaled', (col(target_var) * target_scale))
    df = df.withColumn(target_var+'_standard', ((col(target_var)-target_mean) * target_scale))
    df = df.withColumn(target_var+'_noisy', col(target_var+'_standard') + (randn(seed=noise_seed)*noise_variance))
    means_dict["standardized"] = {
        target_var:{
            "mean": target_mean,
            "scale": target_scale,
            "noise_variance": noise_variance,
        }
    }
    target_var = target_var+'_noisy'

# normalize conditions
means_dict = { "normalized":{}, **means_dict }
if normalize and conditional:
    logger.info(f"Normalizing {conditions} columns")
    # UDF for converting column type from vector to double type
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    # Iterating over columns to be scaled
    for i,cond in enumerate(conditions):
        # convert covariate column to float
        df = df.withColumn(cond, df[cond].cast(DoubleType()))
        # VectorAssembler Transformation - Converting column to vector type
        assembler = VectorAssembler(inputCols=[cond],outputCol=cond+"_vect")
        # MinMaxScaler Transformation
        scaler = MinMaxScaler(inputCol=cond+"_vect", outputCol=cond+"_normed")
        # Pipeline of VectorAssembler and MinMaxScaler
        pipeline = Pipeline(stages=[assembler, scaler])
        # Fitting pipeline on dataframe
        df = pipeline.fit(df).transform(df).withColumn(cond+"_normed", unlist(cond+"_normed")).drop(cond+"_vect")
        imin = df.agg(min(cond)).collect()[0][0]
        imax = df.agg(max(cond)).collect()[0][0]
        means_dict["normalized"] = { **means_dict["normalized"], cond:{ "min":imin, "max":imax } }

# save json file
with open(
    output_path / (filename + ".json"),
    "w",
    encoding="utf-8",
) as f:
    f.write(json.dumps({
        "id": modelid_str,
        "model": conf["model"],
        "data": means_dict,
    }, indent = 4))
logger.info(f"Model info saved to {filename} file.")

# update conditions dict
if normalize and conditional:
    for i,cond in enumerate(conditions):
        conditions[i] = cond+"_normed"

# Check packet_multiply
# Get the first row of the DataFrame for packet multiply
first_row = df.first()
packet_multiply = first_row["packet_multiply"]
logger.info(f"Packet multiply: {packet_multiply}")

# get all latency measurements
latency_samples = df.rdd.map(lambda x: x[target_var]).collect()
latency_samples = np.array(latency_samples)

# get conditions
conditions_data_dict = {}
for condition_str in conditions:
    condition_samples = df.rdd.map(lambda x: x[condition_str]).collect()
    condition_samples = np.array(condition_samples)
    conditions_data_dict[condition_str] = condition_samples

if packet_multiply > 1:
    # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
    latency_samples = latency_samples[packet_multiply-1::packet_multiply]
    for condition_str in conditions:
        condition_data_arr = conditions_data_dict[condition_str]
        conditions_data_dict[condition_str] = condition_data_arr[packet_multiply-1::packet_multiply]

logger.info(f"Number of samples after removing duplicates: {len(latency_samples)}")
logger.info(f"model type: {modelname}")
if recurrent:
    logger.info("building a recurrent model")
    # Create x_train and y_train for recurrent model training
    # Create input (Y) and target (y) data
    Y, y = [], []
    for i in range(len(latency_samples) - recurrent_taps):
        # latency sequence
        Y.append(latency_samples[i:i+recurrent_taps])
        # latency target value
        y.append(latency_samples[i+recurrent_taps])
    Y = np.array(Y).astype(np.float64)
    y = np.array(y) .astype(np.float64)
    # Reshape the input data for LSTM (samples, time steps, features)
    Y = Y.reshape(Y.shape[0], recurrent_taps, 1)
    # randomly select num_training_samples
    indxes = random.sample(range(Y.shape[0]),num_training_samples)
    Y = Y[indxes,:,:]
    y = y[indxes]
    y_train = y
    logger.info(f"Number of sequences y: {len(y)}")
    logger.info(f"shape of target Y: {Y.shape}")

    if conditional:
        logger.info("building a conditional model")
        # Create conditions data
        X_r_dict = {}
        for condition_str in conditions_data_dict:
            X_r_dict[condition_str] = []
            condition_samples = conditions_data_dict[condition_str]
            for i in range(len(latency_samples) - recurrent_taps):
                X_r_dict[condition_str].append(condition_samples[i:i+recurrent_taps])
            X_r_dict[condition_str] = np.array(X_r_dict[condition_str])
            X_r_dict[condition_str] = X_r_dict[condition_str].reshape(
                X_r_dict[condition_str].shape[0], recurrent_taps, 1
            )
            # randomly select num_training_samples (indexes is selected before)
            X_r_dict[condition_str] = X_r_dict[condition_str][indxes,:,:]
        X = np.concatenate(list(X_r_dict.values()), axis=2).astype(np.float64)
        logger.info(f"shape of conditions X: {X.shape}")
        x_train = [Y, X, y]

        if modelname=="gmevm":
            model = ConditionalRecurrentGaussianMixtureEVM(
                centers=centers,
                x_dim=conditions,
                recurrent_taps=recurrent_taps,
                hidden_layers_config=model_config,
            )
        elif modelname=="gmm":
            model = ConditionalRecurrentGaussianMM(
                centers=centers,
                x_dim=conditions,
                recurrent_taps=recurrent_taps,
                hidden_layers_config=model_config,
            )
        else:
            raise Exception("wrong model")
        
    else:
        logger.info("building a non-conditional model")
        x_train = [Y, y]
        if modelname=="gmevm":
            model = RecurrentGaussianMEVM(
                centers=centers,
                recurrent_taps=recurrent_taps,
            )
        elif modelname=="gmm":
            model = RecurrentGaussianMM(
                centers=centers,
                recurrent_taps=recurrent_taps,
            )
        else:
            raise Exception("wrong model")
else:
    logger.info("building a non-recurrent model")
    if conditional:
        logger.info("building a conditional model")
        # create x_train and y_train
        Y = np.array(latency_samples).astype(np.float64)
        X = np.concatenate(list(conditions_data_dict.values()),axis=0).astype(np.float64)

        # select a batch of sequences and a batch of targets, print the result
        indxes = random.sample(range(Y.shape[0]),num_training_samples)
        Y = Y[indxes]
        X = X[indxes]

        logger.info(f"shape of target Y: {Y.shape}")
        logger.info(f"shape of conditions X: {X.shape}")
        x_train = (X,Y)
        y_train = Y

        # create the model
        if modelname=="gmevm":
            model = ConditionalGaussianMixtureEVM(
                x_dim=conditions,
                centers=centers,
                hidden_sizes=model_config
            )
        elif modelname=="gmm":
            model = ConditionalGaussianMM(
                x_dim=conditions,
                centers=centers,
                hidden_sizes=model_config,
            )
        else:
            raise Exception("wrong model")
        
    else:
        logger.info("building a non-conditional model")
        # create x_train and y_train
        Y = np.array(latency_samples).astype(np.float64)
        X = np.zeros(len(Y),dtype=np.float64)

        # select a batch of sequences and a batch of targets, print the result
        indxes = random.sample(range(Y.shape[0]),num_training_samples)
        Y = Y[indxes]
        X = X[indxes]

        logger.info(f"shape of target Y: {Y.shape}")
        logger.info(f"shape of conditions X: {X.shape}")
        x_train = [X,Y]
        y_train = Y

        # create the model
        if modelname=="gmevm":
            model = GaussianMixtureEVM(
                centers=centers
            )
        elif modelname=="gmm":
            model = GaussianMM(
                centers=centers
            )
        else:
            raise Exception("wrong model")

# start training
for idx, round_params in enumerate(training_rounds):
    logger.info(f"training session {idx} with {round_params}")

    # compile
    model.training_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=round_params["learning_rate"],
        ),
        loss=model.loss,
    )
    # train!

    model.training_model.fit(
        x=x_train,
        y=y_train,
        steps_per_epoch=len(y_train) // batch_size,
        epochs=round_params["epochs"],
        verbose=1,
    )

# training done, save the model
model.save(output_path / (filename+".h5"))
logger.success("Model saved successfully.")


