import getopt
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import os
import signal
import sys
import warnings
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from petastorm.spark import SparkDatasetConverter
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

warnings.filterwarnings("ignore")
# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_train_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:l:c:e:",
            ["dataset=", "label=", "config=", "ensembles="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark train -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark train "
                + "-d <dataset label> -l <label> -c <config json file>",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-c", "--config-file"):
            with open(arg) as json_file:
                data = json.load(json_file)
            args_dict["train_config"] = data
        elif opt in ("-e", "--ensembles"):
            args_dict["ensembles"] = int(arg)

    return args_dict


def run_train_processes(exp_args: list):
    logger.info(
        "Prepare models benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    # project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = str(p) + "/" + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)
    parquet_folder = main_path + "__trainparquets__"
    os.makedirs(parquet_folder, exist_ok=True)

    train_configs = exp_args["train_config"]

    n_workers = 18
    logger.info(f"Initializng {n_workers} workers")
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # create params list for each run
    n_runs = len(train_configs.keys()) * exp_args["ensembles"]
    params_list = []
    for ensemble_num in range(exp_args["ensembles"]):
        for model_conf_key in train_configs.keys():
            model_conf = train_configs[model_conf_key]

            # create and prepare the results directory
            records_path = project_path + model_conf_key + "/"
            os.makedirs(records_path, exist_ok=True)

            # save the json info
            jsoninfo = json.dumps(model_conf)
            with open(records_path + "info.json", "w") as f:
                f.write(jsoninfo)

            dataset_path = main_path + exp_args["dataset"] + "_results/"
            params = {
                "ensembles": exp_args["ensembles"],
                "order_seed": 9988334,
                "ensemble_num": ensemble_num,
                "sample_seed": ensemble_num * 101012,
                "dataset_path": dataset_path,
                "records_path": records_path,
                "model_conf": model_conf,
                "model_conf_key": model_conf_key,
                "spark_total_memory": "70g",
                "spark_subprocess_memory": "3g",
                "parquet_folder": parquet_folder,
            }
            params_list.append(params)

    # load dataset and sample
    load_dataset_and_sample(params)

    try:
        logger.info(f"Starting {n_runs} jobs")
        res = pool.map_async(train_model, params_list)
        logger.info("Waiting for results")
        res.get(100)  # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal termination")
        pool.close()


def load_dataset_and_sample(params):
    # load the dataset, take 'dataset' 'ensembles' times
    # save them in parquet files, send the address to the
    # sub-process

    logger.info("Load dataset and sample")

    # init Spark
    spark = (
        SparkSession.builder.appName("Training")
        .config("spark.driver.memory", params["spark_total_memory"])
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )
    # sc = spark.sparkContext

    # Set a cache directory on DBFS FUSE for intermediate data.
    file_path = dirname(abspath(__file__))
    spark_cash_addr = "file://" + file_path + "/__sparkcache__/__main__"
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, spark_cash_addr)
    logger.info(
        f"load_dataset_and_sample: Spark cache folder is set up at: {spark_cash_addr}"
    )

    # find records_paths
    records_path = params["dataset_path"]

    # read all the files from the project
    files = []
    logger.info(f"Opening the path '{records_path}'")
    all_files = os.listdir(records_path)
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + "/" + f)

    # read all files into one Spark df
    main_df = spark.read.parquet(*files)

    # Absolutely necessary for randomizing the rows (bug fix)
    # first shuffle, then sample!
    main_df = main_df.orderBy(rand(seed=params["order_seed"]))
    training_params = params["model_conf"]["training_params"]

    for ensemble_num in range(params["ensembles"]):

        # take the desired number of records for learning
        df_train = main_df.sample(
            withReplacement=False,
            fraction=training_params["dataset_size"] / main_df.count(),
            seed=params["sample_seed"],
        )

        logger.info(
            f"{ensemble_num}: sample {training_params['dataset_size']} rows, result {df_train.count()} samples"
        )

        ensemble_parquet_addr = params["parquet_folder"] + f"/{ensemble_num}.parquet"
        logger.info(f"{ensemble_num}: writing sub-dataset into {ensemble_parquet_addr}")
        pandas_df = df_train.toPandas()
        pandas_df.to_parquet(ensemble_parquet_addr, compression="snappy")
        del df_train
        del pandas_df


def train_model(params):

    import tensorflow as tf
    from petastorm import TransformSpec
    from petastorm.spark import SparkDatasetConverter, make_spark_converter
    from pr3d.de import ConditionalGammaMixtureEVM, ConditionalGaussianMM
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import rand

    logger.info(
        f"{params['model_conf_key']}.{params['ensemble_num']}: starts with params {params}"
    )

    # set params
    model_conf = params["model_conf"]
    module_label = params["model_conf_key"]
    predictors_path = params["records_path"]
    training_params = model_conf["training_params"]

    # set data types
    # npdtype = np.float64
    # tfdtype = tf.float64
    strdtype = "float64"

    logger.info(f"Opening predictors directory '{predictors_path}'")
    os.makedirs(predictors_path, exist_ok=True)

    # read sub-dataset into Pandas df
    ensemble_parquet_addr = (
        params["parquet_folder"] + f"/{params['ensemble_num']}.parquet"
    )
    df_train = pd.read_parquet(ensemble_parquet_addr)
    logger.info(
        f"{module_label}.{params['ensemble_num']}: dataset loaded, train sampels: {len(df_train)}"
    )

    condition_labels = model_conf["condition_labels"]

    # get parameters
    y_label = model_conf["y_label"]
    model_type = model_conf["type"]
    condition_labels = model_conf["condition_labels"]
    training_rounds = training_params["rounds"]
    batch_size = training_params["batch_size"]

    # dataset pre process
    df_train = df_train[[y_label, *condition_labels]]
    df_train["y_input"] = df_train[y_label]
    df_train = df_train.drop(columns=[y_label])

    # initiate the non conditional predictor
    if model_type == "gmm":
        model = ConditionalGaussianMM(
            x_dim=condition_labels,
            centers=model_conf["centers"],
            hidden_sizes=model_conf["hidden_sizes"],
            dtype=strdtype,
            bayesian=model_conf["bayesian"],
            # batch_size = 1024,
        )
    elif model_type == "gmevm":
        model = ConditionalGammaMixtureEVM(
            x_dim=condition_labels,
            centers=model_conf["centers"],
            hidden_sizes=model_conf["hidden_sizes"],
            dtype=strdtype,
            bayesian=model_conf["bayesian"],
            # batch_size = 1024,
        )

    X = df_train[condition_labels]
    Y = df_train.y_input

    steps_per_epoch = len(df_train) // batch_size

    for idx, round_params in enumerate(training_rounds):

        logger.info(
            f"{module_label}.{params['ensemble_num']}: training session "
            + f"{idx+1}/{len(training_rounds)} with {round_params}, "
            + f"steps_per_epoch: {steps_per_epoch}, batch size: {batch_size}"
        )

        model.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=round_params["learning_rate"]
            ),
            loss=model.loss,
        )

        Xnp = np.array(X)
        Ynp = np.array(Y)
        training_data = tuple([Xnp[:, i] for i in range(len(condition_labels))]) + (
            Ynp,
        )

        model.training_model.fit(
            x=training_data,
            y=Ynp,
            steps_per_epoch=steps_per_epoch,
            epochs=round_params["epochs"],
            verbose=0,
        )

    # training done, save the model
    model.save(predictors_path + f"model_{params['ensemble_num']}.h5")
    with open(
        predictors_path + f"model_{params['ensemble_num']}.json", "w"
    ) as write_file:
        json.dump(model_conf, write_file, indent=4)

    logger.info(
        f"{model_type} {'bayesian' if model.bayesian else 'non-bayesian'} "
        + f"model {params['ensemble_num']} got trained and saved."
    )
