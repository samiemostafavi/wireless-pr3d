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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pr3d.de import ConditionalGammaMixtureEVM, ConditionalGaussianMM
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")


def parse_plot_parquet_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:p:l:t:",
            [
                "dataset=",
                "parquet=",
                "label=",
                "target="
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

    return args_dict

def lookup_df(folder_path, parquet, spark):
    parquet_path = os.path.join(folder_path, f"{parquet}.parquet")
    df = spark.read.parquet(parquet_path)
    total_count = df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return df

def run_plot_parquet_processes(exp_args: list):
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

    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))

    logger.info(f"Plotting dataframe")
    total_count = df.count()

    measurements = df.rdd.map(lambda x: x[exp_args["target"]]).collect()

    # plot measurements
    nparr = np.array(measurements)
    ax.bar(
        range(total_count),
        nparr/1e6,
        label="delay measurements",
        width=5,
        snap=False
    )

    ax.set_title(f"title")
    ax.set_xlabel("time index")
    ax.set_ylabel("delay [ms]")
    ax.grid()
    ax.legend()

    # pdf figure
    fig.tight_layout()
    fig.savefig(project_path + f"{exp_args['parquet']}_timeseries.png")