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
from os import environ
import math
import numpy as np
from loguru import logger
#from pyspark.sql import SparkSession
from pr3d.de import ConditionalRecurrentGaussianMM,ConditionalGaussianMM,ConditionalRecurrentGaussianMixtureEVM,RecurrentGaussianMM,RecurrentGaussianMEVM,GaussianMM,ConditionalGaussianMixtureEVM,GaussianMixtureEVM
from scipy.stats import norm
from scipy.optimize import root_scalar
from pyspark.sql.functions import col, mean, randn, min, max, udf, isnan, isnull
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pathlib import Path
from pyspark.sql import SparkSession

# opening evaluation configuration
if environ.get('CONF_FILE_ADDR') is not None:
    with open(environ.get('CONF_FILE_ADDR')) as json_file:
        conf = json.load(json_file)
else:
    raise Exception("CONF_FILE_ADDR environment variable is not set.")

parquet_files = conf["data"]["parquet_files"]
model_h5_addr = conf["model"]["h5_addr"]
model_json_addr = conf["model"]["json_addr"]
test_cases_num = conf["data"]["test_cases_num"]
quantile_levels = conf["data"]["quantile_levels"]

# opening model and data configuration
with open(model_json_addr) as json_file:
    loaded_dict = json.load(json_file)
    modelid_str = loaded_dict["id"]
    model_info = loaded_dict["model"]
    model_data_info = loaded_dict["data"]

output_dir = conf["data"]["output_dir"]
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
filename = "result_" + modelid_str

target_var = model_info["target_var"]
recurrent = (True if model_info["recurrent_taps"] > 0 else False)
conditions = model_info["conditions"]
conditional = (True if len(conditions) > 0 else False)


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
standardize = (True if model_data_info["standardized"] else False)
if standardize:
    logger.info(f"Standardizing {target_var} column")
    noise_seed = random.randint(0,1000)
    target_mean = model_data_info["standardized"][target_var]["mean"]
    target_scale = model_data_info["standardized"][target_var]["scale"]
    df = df.withColumn(target_var+'_scaled', (col(target_var) * target_scale))
    df = df.withColumn(target_var+'_standard', ((col(target_var)-target_mean) * target_scale))
    target_var = target_var+'_standard'

# normalize conditions
normalize = (True if model_data_info["normalized"] else False)
if normalize and conditional:
    logger.info(f"Normalizing {list(model_data_info['normalized'].keys())} columns")

    # Iterating over columns to be scaled
    for cond in model_data_info["normalized"]:
        min_value = model_data_info["normalized"][cond]["min"]
        max_value = model_data_info["normalized"][cond]["max"]

        # UDF for scaler function
        def custom_scaler(value):
            return (value - min_value) / (max_value - min_value)

        # Register the UDF
        myscalar = udf(custom_scaler, DoubleType())

        # convert covariate column to float
        df = df.withColumn(cond, df[cond].cast(DoubleType()))
        df = df.withColumn(cond+"_normed", myscalar(col(cond)))

        index = conditions.index(cond)
        conditions[index] = cond+"_normed"

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

# load the model
logger.info(f"loading model type: {model_info['modelname']}")
if recurrent:
    logger.info("loading a recurrent model")

    if conditional:
        logger.info("loading a conditional model")
        if model_info["modelname"] == "gmm":
            model = ConditionalRecurrentGaussianMM(h5_addr=model_h5_addr)
        elif model_info["modelname"] == "gmevm":
            model = ConditionalRecurrentGaussianMixtureEVM(h5_addr=model_h5_addr)
        else:
            raise Exception("wrong model")
    else:
        logger.info("loading a non-conditional model")
        if model_info["modelname"] == "gmm":
            model = RecurrentGaussianMM(h5_addr=model_h5_addr)
        elif model_info["modelname"] == "gmevm":
            model = RecurrentGaussianMEVM(h5_addr=model_h5_addr)
        else:
            raise Exception("wrong model")
        
    logger.success(f"model loaded, number of taps: {model.recurrent_taps}")
    recurrent_taps = model.recurrent_taps

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
    indxes = random.sample(range(Y.shape[0]),test_cases_num)
    Y = Y[indxes,:,:]
    y = y[indxes]

    logger.info(f"Number of sequences y: {len(y)}")
    logger.info(f"shape of target Y: {Y.shape}")

    if conditional:
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

else:
    logger.info("loading a non-recurrent model")
    if conditional:
        logger.info("loading a conditional model")
        if model_info["modelname"] == "gmm":
            model = ConditionalGaussianMM(h5_addr=model_h5_addr)
        elif model_info["modelname"] == "gmevm":
            model = ConditionalGaussianMixtureEVM(h5_addr=model_h5_addr)
        else:
            raise Exception("wrong model")
        
        # create x_train and y_train
        Y = np.array(latency_samples).astype(np.float64)
        X = np.concatenate(list(conditions_data_dict.values()),axis=0).astype(np.float64)

    else:
        logger.info("loading a non-conditional model")
        if model_info["modelname"] == "gmm":
            model = GaussianMM(h5_addr=model_h5_addr)
        elif model_info["modelname"] == "gmevm":
            model = GaussianMixtureEVM(h5_addr=model_h5_addr)
        else:
            raise Exception("wrong model")
        
        # create x_train and y_train
        Y = np.array(latency_samples).astype(np.float64)
        X = np.zeros(len(Y),dtype=np.float64)

    # select a batch of sequences and a batch of targets, print the result
    indxes = random.sample(range(Y.shape[0]),test_cases_num)
    Y = Y[indxes]
    X = X[indxes]

    logger.info(f"shape of target Y: {Y.shape}")
    logger.info(f"shape of conditions X: {X.shape}")

model.core_model._model.summary()

def find_quantile_rec_model(rmodel, Y, X, quantile_level, lower_bound=-20, upper_bound=60):
    prediction_res = rmodel._params_model.predict(
        [Y,X]
    )
    weights = np.array(prediction_res[0])
    locs = np.array(prediction_res[1])
    scales = np.array(prediction_res[2])

    # Calculate the quantile values using numerical root-finding
    def objective(x,weights,locs,scales):
        cdf_values = np.sum(weights[:, np.newaxis] * norm.cdf(x, loc=locs[:, np.newaxis], scale=scales[:, np.newaxis]), axis=0)
        return cdf_values - quantile_level

    quantile_values = []
    for i in range(len(weights)):
        # Use root_scalar to find the quantile value that satisfies the objective
        result = root_scalar(
            objective,
            bracket=[lower_bound, upper_bound],
            args=(weights[i], locs[i], scales[i]),
            xtol=1e-2,
        )
        quantile_values.append(result.root)

    return np.array(quantile_values)


def find_quantile_nonrec_model(model, X, quantile_level, lower_bound=-20, upper_bound=60):
    prediction_res = model._params_model.predict(
        {"input":X}
    )
    weights = np.array(prediction_res[0])
    locs = np.array(prediction_res[1])
    scales = np.array(prediction_res[2])

    # Calculate the quantile values using numerical root-finding
    def objective(x,weights,locs,scales):
        cdf_values = np.sum(weights[:, np.newaxis] * norm.cdf(x, loc=locs[:, np.newaxis], scale=scales[:, np.newaxis]), axis=0)
        return cdf_values - quantile_level

    quantile_values = []
    for i in range(len(weights)):
        # Use root_scalar to find the quantile value that satisfies the objective
        result = root_scalar(
            objective,
            bracket=[lower_bound, upper_bound],
            args=(weights[i], locs[i], scales[i]),
            xtol=1e-2,
        )
        quantile_values.append(result.root)

    return np.array(quantile_values)

results_dict = {}
for quantile_level in quantile_levels:
    # baseline
    logger.info("Baseline: {:1.2e}".format(1-quantile_level))

    # calc quantile rmdn
    if recurrent:
        quantile_vals = find_quantile_rec_model(model,Y,X,quantile_level)
        res = np.sum(y > quantile_vals)/float(test_cases_num)
        logger.info("result: {:1.2e}".format(res))
        results_dict[str(quantile_level)] = res
    else:
        # calc quantile mdn
        quantile_vals = find_quantile_nonrec_model(model,X,quantile_level)
        res = np.sum(Y > quantile_vals)/float(test_cases_num)
        logger.info("result: {:1.2e}".format(res))
        results_dict[str(quantile_level)] = res

# save json file
with open(
    output_path / (filename + ".json"),
    "w",
    encoding="utf-8",
) as f:
    f.write(json.dumps({
        "id": modelid_str,
        "result": results_dict,
        **conf
    }, indent = 4))

logger.info(f"Evaluation results saved to {filename} file.")