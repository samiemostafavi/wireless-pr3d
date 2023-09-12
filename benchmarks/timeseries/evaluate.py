import os
from os import environ
from loguru import logger
import builtins

CPU_ONLY = False
if environ.get('CPU_ONLY') is not None:
    val = environ.get('CPU_ONLY')
    if val.lower()=='true':
        logger.info("CPU_ONLY is set")
        CPU_ONLY = True

if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import random, json
from os import environ

import multiprocessing as mp
import multiprocessing.context as ctx
# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")

import numpy as np
import signal
from loguru import logger
from scipy.stats import norm
from scipy.optimize import root_scalar
from pyspark.sql.functions import col, mean, randn, min, max, udf, isnan, isnull
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pr3d.common.core import NonConditionalDensityEstimator,NonConditionalRecurrentDensityEstimator
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pathlib import Path
from pyspark.sql import SparkSession
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

def find_quantile_cond_rec_model(rmodel, Y, X, quantile_level, lower_bound=-20, upper_bound=60):
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


def find_quantile_cond_nonrec_model(model, conditions_data_dict, quantile_level, lower_bound=-20, upper_bound=60):
    prediction_res = model._params_model.predict(
        conditions_data_dict
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


def find_quantile_noncond_rec_model(model : NonConditionalRecurrentDensityEstimator, Y, quantile_level, lower_bound=-20, upper_bound=100):

    # Calculate the quantile values using numerical root-finding
    def objective(x,newY):
        x = np.array([x,x])
        pdf, logpdf, cdf = model.prob_batch(newY,x)
        return cdf[0] - quantile_level

    results = []
    for rowY in Y:
        newY = np.squeeze(rowY).tolist()
        newY = np.array([newY,newY])
        
        # Use root_scalar to find the quantile value that satisfies the objective
        result = root_scalar(
            objective,
            bracket=[lower_bound, upper_bound],
            args=(newY),
            xtol=1e-2,
        )
        result = result.root
        print(result)

        pdf, logpdf, cdf = model.prob_batch(newY,np.array([result,result]))
        print(f"quantile_level: {quantile_level}, result: {cdf[0]}")
        results.append(result)

    return results

def new_find_quantile_noncond_rec_model(model : NonConditionalRecurrentDensityEstimator, Y, quantile_level, lower_bound=-10, upper_bound=40, divisions=128):

    def find_quantile(rowY,x):
        newY = np.expand_dims(rowY,axis=0)
        newY = newY.tolist()
        newY = np.repeat(newY, len(x), axis=0)
        pdf, logpdf, cdf = model.prob_batch(newY,x,batch_size=len(x))
        def positivediff(inp):
            if quantile_level - inp >= 0:
                return quantile_level - inp
            else:
                return 100000
        def negativediff(inp):
            if quantile_level - inp >= 0:
                return 100000
            else:
                return inp - quantile_level
        
        cdf = cdf.tolist()
        cdfneg = builtins.min(cdf, key=positivediff)
        resultneg = x[cdf.index(cdfneg)]
        cdfpos = builtins.min(cdf, key=negativediff)
        resultpos = x[cdf.index(cdfpos)]
        return resultneg, resultpos

    results = []
    for rowY in Y:
        x = np.linspace(start=lower_bound,stop=upper_bound,num=divisions)
        resultneg,resultpos = find_quantile(rowY,x)
        #x = np.linspace(start=resultneg,stop=resultpos,num=divisions)
        #resultneg,resultpos = find_quantile(rowY,x)
        #print(resultneg)
        #print(resultpos)
        
        result = (resultpos + resultneg)/2.0
        results.append(result)

        #newY = np.squeeze(rowY).tolist()
        #newY = np.array([newY,newY])
        #pdf, logpdf, cdf = model.prob_batch(newY,np.array([result,result]))
        #print(f"result: {result}, quantile_level: {quantile_level}, result: {cdf[0]}")

    return results

def find_quantile_noncond_nonrec_model(model : NonConditionalDensityEstimator, quantile_level, lower_bound=-20, upper_bound=60):

    # Calculate the quantile values using numerical root-finding
    def objective(x):
        pdf, logpdf, cdf = model.prob_batch(np.expand_dims(np.array([x,x]),axis=1))
        return cdf[0] - quantile_level

    # Use root_scalar to find the quantile value that satisfies the objective
    result = root_scalar(
        objective,
        bracket=[lower_bound, upper_bound],
        xtol=1e-2,
    )
    result = result.root
    #print(result)

    #pdf, logpdf, cdf = model.prob_batch(np.expand_dims(np.array([result,result]),axis=1))
    #print(f"quantile_level: {quantile_level}, result: {cdf[0]}")

    return result

def eval_model(process_inp):
    from pr3d.de import ConditionalRecurrentGaussianMM,ConditionalGaussianMM,ConditionalRecurrentGaussianMixtureEVM,RecurrentGaussianMM,RecurrentGaussianMEVM,GaussianMM,ConditionalGaussianMixtureEVM,GaussianMixtureEVM

    recurrent = process_inp["recurrent"]
    model_info = process_inp["model_info"]
    model_h5_addr = process_inp["model_h5_addr"]
    conditional = process_inp["conditional"]
    quantile_level = process_inp["quantile_level"]
    latency_samples = process_inp["latency_samples"]
    conditions_data_dict = process_inp["conditions_data_dict"]
    test_cases_num = process_inp["test_cases_num"]

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
            # select a batch of sequences and a batch of targets, print the result
            indxes = random.sample(range(Y.shape[0]),test_cases_num)
            #X = np.concatenate(list(conditions_data_dict.values()),axis=0).astype(np.float64)
            Y = Y[indxes]
            X = {}
            for cond in conditions_data_dict:
                X[cond] = np.array(conditions_data_dict[cond][indxes]).astype(np.float64)

            logger.info(f"shape of target Y: {Y.shape}")
            logger.info(f"shape of conditions X: {X.keys()}, X[0]:{next(iter(X.values())).shape}")

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

    #model.core_model._model.summary()

    # baseline
    logger.info("starting for quantile: {:1.2e}".format(1-quantile_level))

    # calc quantile rmdn
    if recurrent:
        if conditional:
            quantile_vals = find_quantile_cond_rec_model(model,Y,X,quantile_level)
        else:
            quantile_vals = new_find_quantile_noncond_rec_model(model,Y,quantile_level)

        #print(y)
        #print(quantile_vals)
        res = np.sum(y > quantile_vals)/float(test_cases_num)
        logger.info("quantile: {:1.2e}, result: {:1.2e}".format(1-quantile_level, res))

    else:
        if conditional:
            # calc quantile mdn
            quantile_vals = find_quantile_cond_nonrec_model(model,X,quantile_level)
        else:
            quantile_vals = find_quantile_noncond_nonrec_model(model,quantile_level)

        res = np.sum(Y > quantile_vals)/float(test_cases_num)
        logger.info("quantile: {:1.2e}, result: {:1.2e}".format(1-quantile_level, res))

    return res

if __name__ == '__main__':

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
    resultid_str = get_yymmddhhmmss()
    filename = "result_" + resultid_str

    orig_target_var = model_info["target_var"]
    recurrent = (True if model_info["recurrent_taps"] > 0 else False)
    orig_conditions = model_info["conditions"]
    conditional = (True if len(orig_conditions) > 0 else False)
    parallelize = conf["data"]["parallelize"]

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
        (col(orig_target_var).cast("double").isNotNull()) &
        ~isnan(orig_target_var) &
        ~isnull(orig_target_var)
    )

    # preprocess add noise and standardize
    standardize = (True if model_data_info["standardized"] else False)
    if standardize:
        logger.info(f"Standardizing {orig_target_var} column")
        noise_seed = random.randint(0,1000)
        target_mean = model_data_info["standardized"][orig_target_var]["mean"]
        target_scale = model_data_info["standardized"][orig_target_var]["scale"]
        df = df.withColumn(orig_target_var+'_scaled', (col(orig_target_var) * target_scale))
        df = df.withColumn(orig_target_var+'_standard', ((col(orig_target_var)-target_mean) * target_scale))
        target_var = orig_target_var+'_standard'

    # normalize conditions
    conditions = []
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

            conditions.append(cond+"_normed")

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
        condition_samples = np.array(condition_samples).astype(np.float64)
        conditions_data_dict[condition_str] = condition_samples

    if packet_multiply > 1:
        # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
        latency_samples = latency_samples[packet_multiply-1::packet_multiply]
        for condition_str in conditions:
            condition_data_arr = conditions_data_dict[condition_str]
            conditions_data_dict[condition_str] = condition_data_arr[packet_multiply-1::packet_multiply]

    logger.info(f"Number of samples after removing duplicates: {len(latency_samples)}")

    # prepare list of dicts to pass to the workers
    dict_arr = []
    for quantile_level in quantile_levels:
        process_inp = {}
        process_inp["recurrent"] = recurrent
        process_inp["model_info"] = model_info
        process_inp["model_h5_addr"] = model_h5_addr
        process_inp["conditional"] = conditional
        process_inp["quantile_level"] = quantile_level
        process_inp["latency_samples"] = latency_samples
        process_inp["conditions_data_dict"] = conditions_data_dict
        process_inp["test_cases_num"] = test_cases_num
        dict_arr.append(process_inp)

    if parallelize:
        n_workers = len(quantile_levels)
        logger.info(f"Initializng {n_workers} worker(s)")
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = mp.Pool(n_workers)
        signal.signal(signal.SIGINT, original_sigint_handler)

        n_runs = n_workers
        try:
            logger.info(f"Starting {n_runs} jobs")
            results = pool.map_async(eval_model, dict_arr)
            logger.info("Waiting for results")
            results = results.get(100000)  # Without the timeout this blocking call ignores all signals.
        except KeyboardInterrupt:
            logger.info("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        else:
            logger.info("Normal termination")
            pool.close()

        # collect the results from the workers
        results_dict = {}
        for idx, result in enumerate(results):
            results_dict[str(quantile_levels[idx])] = result
    else:
        n_workers = 1
        results_dict = {}
        for idx,quantile in enumerate(quantile_levels):
            result = eval_model(dict_arr[idx])
            results_dict[str(quantile_levels[idx])] = result



    # save json file
    with open(
        output_path / (filename + ".json"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps({
            "id": resultid_str,
            "result": results_dict,
            "data": conf["data"],
            "model":{
                "id":modelid_str,
                **conf["model"]
            }
        }, indent = 4))

    logger.info(f"Evaluation results saved to {filename} file.")