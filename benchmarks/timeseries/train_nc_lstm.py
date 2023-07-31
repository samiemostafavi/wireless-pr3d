import ast
import getopt
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from os.path import abspath, dirname
from pathlib import Path
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pr3d.de import ConditionalGammaMixtureEVM, ConditionalGaussianMM
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")


def parse_train_nc_lstm_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:p:l:t:s:a:",
            [
                "dataset=",
                "parquet=",
                "label=",
                "target=",
                "taps=",
                "num_samples="
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark validate "
                + "-q <qlens> -d <dataset> -m <trained models> -l <label> -e <ensemble num>",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-p", "--parquet"):
            args_dict["parquet"] = arg
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-t", "--target"):
            args_dict["target"] = arg
        elif opt in ("-s", "--taps"):
            args_dict["taps"] = int(arg)
        elif opt in ("-a", "--samples"):
            args_dict["num_samples"] = int(arg)

    return args_dict

def lookup_df(folder_path, parquet, spark):
    parquet_path = os.path.join(folder_path, f"{parquet}.parquet")
    df = spark.read.parquet(parquet_path)
    total_count = df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return df

def run_train_nc_lstm_processes(exp_args: list):
    logger.info(
        "Prepare models benchmark validate args "
        + f"with command line args: {exp_args}"
    )

    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )


    # this project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    df = lookup_df(dataset_project_path,exp_args["parquet"],spark)

    measurements = df.rdd.map(lambda x: x[exp_args["target"]]).collect()
    time_series_data = np.array(measurements)/1e6

    # limit the number of samples
    time_series_data = time_series_data[:exp_args["num_samples"]]
    logger.info(f"Limited the number of samples for training: {len(time_series_data)}")

    # number of taps
    num_taps = exp_args["taps"]

    # Create input (X) and target (y) data
    X, y = [], []
    for i in range(len(time_series_data) - num_taps):
        X.append(time_series_data[i:i+num_taps])
        y.append(time_series_data[i+num_taps])
    X = np.array(X)
    y = np.array(y)

    # Reshape the input data for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], num_taps, 1)

    # Split the data into training and testing sets (adjust the split ratio as needed)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build the autoregressive LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(num_taps, 1)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    # Save the model to a file
    model.save("autoregressive_model.h5")
    print("Model saved successfully.")
