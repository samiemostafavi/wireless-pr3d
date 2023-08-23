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
from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pr3d.de import ConditionalGammaMixtureEVM, ConditionalGaussianMM
from pyspark.sql import SparkSession
from pyspark.sql.functions import first
from pyspark.sql.window import Window
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")


def parse_plot_acf_args(argv: list[str]):

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

def run_plot_acf_processes(exp_args: list):
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

    # Check packet_multiply
    # Get the first row of the DataFrame for packet multiply
    first_row = df.first()
    packet_multiply = first_row["packet_multiply"]
    logger.info(f"Packet multiply: {packet_multiply}")

    # get all measurements
    measurements = df.rdd.map(lambda x: x[exp_args["target"]]).collect()
    time_series_data = np.array(measurements)/1e6

    # limit the number of samples
    time_series_data = time_series_data[:exp_args["num_samples"]]

    if packet_multiply > 1:
        # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
        time_series_data = time_series_data[packet_multiply-1::packet_multiply]

    logger.info(f"Limited the number of samples for autocorrelation: {len(time_series_data)}")

    # number of taps
    num_taps = exp_args["taps"]

    # Visualize autocorrelation
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))
    logger.info(f"Plotting ACF")

    plot_acf(x=time_series_data, ax=ax, lags=num_taps)

    ax.set_title(f"Autocorrelation")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid()

    # pdf figure
    fig.tight_layout()
    fig.savefig(project_path + f"{exp_args['parquet']}_acf.png")