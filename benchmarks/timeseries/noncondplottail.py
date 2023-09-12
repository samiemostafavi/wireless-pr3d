import ast
import getopt
import json
import multiprocessing as mp
import os
import sys
import time
import itertools
import polars
import warnings
from os.path import abspath, dirname
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pyspark.sql import SparkSession

import random, json
import numpy as np
import signal
from loguru import logger
from scipy.stats import norm
from scipy.optimize import root_scalar
from pyspark.sql.functions import col, mean, randn, min, max, udf, isnan, isnull
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pathlib import Path
from pyspark.sql import SparkSession
import pandas as pd
from pr3d.de import ConditionalRecurrentGaussianMM,ConditionalGaussianMM,ConditionalRecurrentGaussianMixtureEVM,RecurrentGaussianMM,RecurrentGaussianMEVM,GaussianMM,ConditionalGaussianMixtureEVM,GaussianMixtureEVM
from matplotlib.lines import Line2D

import scienceplots
plt.style.use(['science','ieee'])

from datetime import datetime

spark = (
    SparkSession.builder.master("local")
    .appName("LoadParquets")
    .config("spark.executor.memory", "6g")
    .config("spark.driver.memory", "70g")
    .config("spark.driver.maxResultSize", 0)
    .getOrCreate()
)

parquet_groups = [
    "timeseries/newr1ep5g_results/s1/10-42-3-2_55500_20230809_114214.parquet",
    "timeseries/newr1ep5g_results/s1/10-42-3-2_55500_20230809_121110.parquet",
    "timeseries/newr1ep5g_results/s1/10-42-3-2_55500_20230809_124005.parquet",
    "timeseries/newr1ep5g_results/s1/10-42-3-2_55500_20230809_130901.parquet",
    "timeseries/newr1ep5g_results/s1/10-42-3-2_55500_20230809_133756.parquet",
    "timeseries/newr1ep5g_results/s1/10-42-3-2_55500_20230809_140652.parquet",
    "timeseries/newr1ep5g_results/s10/10-70-70-3_55500_20230816_110947.parquet",
    "timeseries/newr1ep5g_results/s10/10-70-70-3_55500_20230816_113841.parquet",
    "timeseries/newr1ep5g_results/s10/10-70-70-3_55500_20230816_120736.parquet",
    "timeseries/newr1ep5g_results/s10/10-70-70-3_55500_20230816_123630.parquet",
    "timeseries/newr1ep5g_results/s10/10-70-70-3_55500_20230816_130524.parquet",
    "timeseries/newr1ep5g_results/s10/10-70-70-3_55500_20230816_133419.parquet",
]
parquet_labels = [
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]
darker = 0.9
parquet_color_pallete = [
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.9*darker,0.4*darker,0.0*darker, 1],
    [0.9*darker,0.4*darker,0.0*darker, 1],
    [0.9*darker,0.4*darker,0.0*darker, 1],
    [0.9*darker,0.4*darker,0.0*darker, 1],
    [0.9*darker,0.4*darker,0.0*darker, 1],
    [0.9*darker,0.4*darker,0.0*darker, 1],
]
parquet_line = '-'

custom_legend_items = ['Uplink', 'Downlink']
custom_legend_lines = [Line2D([0], [0], color=[0.1*darker,0.2*darker,0.8*darker, 1], lw=4),
                Line2D([0], [0], color=[0.9*darker,0.4*darker,0.0*darker, 1], lw=4)]


result_file = "timeseries/newr1ep5g_results/expecapaper.png"


model_h5_addr = "timeseries/newr1ep5g_results/s1s2s3/trained_models/model_230905100956_099812.h5"
model_json_addr = "timeseries/newr1ep5g_results/s1s2s3/trained_models/model_230905100956_099812.json"
plot_model = False

if plot_model:
    # opening model and data configuration
    with open(model_json_addr) as json_file:
        loaded_dict = json.load(json_file)
        modelid_str = loaded_dict["id"]
        model_info = loaded_dict["model"]
        model_data_info = loaded_dict["data"]

tail_fig, tail_ax = plt.subplots(nrows=1, ncols=1,  figsize=(7/2.5, 5/2.5))
for idx,parquet_files in enumerate(parquet_groups):

    df = spark.read.parquet(parquet_files)
    total_count = df.count()
    logger.info(f"Parquet files {parquet_files} are loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    #change column name, replace . with _
    column_names = []
    for mcol in df.columns:
        newcol = mcol.replace(".", "_")
        df = df.withColumnRenamed(mcol, newcol)
        column_names.append(newcol)
    logger.info(f"column names: {column_names}")

    key_label = "delay_send"
    target_scale = 1e-6
    y_points = [2.5,100,18]

    first_timestamp=df.head().timestamps_client_send_wall
    # Convert nanoseconds to seconds
    timestamp_s = int(first_timestamp) / 1e9  # 1e9 nanoseconds = 1 second

    # Convert to datetime object
    dt_object = datetime.fromtimestamp(timestamp_s)

    # Print the datetime object in a human-readable format
    logger.info(f"Date and Time of recording this dataset: {dt_object.strftime('%Y-%m-%d %H:%M:%S')}")

    # remove non latency values
    df = df.filter(
        (col(key_label).cast("double").isNotNull()) &
        ~isnan(key_label) &
        ~isnull(key_label)
    )

    if plot_model:
        # preprocess add noise and standardize
        standardize = (True if model_data_info["standardized"] else False)
        if standardize:
            logger.info(f"Standardizing {key_label} column")
            noise_seed = random.randint(0,1000)
            target_mean = model_data_info["standardized"][key_label]["mean"]
            target_scale = model_data_info["standardized"][key_label]["scale"]
            df = df.withColumn(key_label+'_scaled', (col(key_label) * target_scale))
            df = df.withColumn(key_label+'_standard', ((col(key_label)-target_mean) * target_scale))
            key_label = key_label+'_scaled'
            key_label_standard = key_label+'_standard'
        else:
            # preprocess add noise and standardize
            logger.info(f"Scaling {key_label} column")
            noise_seed = random.randint(0,1000)
            df = df.withColumn(key_label+'_scaled', (col(key_label) * target_scale))
            key_label = key_label+'_scaled'
    else:
        # preprocess add noise and standardize
        logger.info(f"Scaling {key_label} column")
        noise_seed = random.randint(0,1000)
        df = df.withColumn(key_label+'_scaled', (col(key_label) * target_scale))
        key_label = key_label+'_scaled'


    # Check packet_multiply
    # Get the first row of the DataFrame for packet multiply
    first_row = df.first()
    packet_multiply = first_row["packet_multiply"]
    logger.info(f"Packet multiply: {packet_multiply}")

    # get all latency measurements
    df = pd.DataFrame(df.rdd.map(lambda x: x[key_label]).collect())

    if packet_multiply > 1:
        # Skip the first (packet_multiply-1) samples, then take every packet_multiply samples from the time_series_data
        df = df[packet_multiply-1::packet_multiply]

    logger.info(f"Number of samples after removing duplicates: {len(df)}")

    # bulk plot axis
    y_points = np.linspace(
        start=y_points[0],
        stop=y_points[2],
        num=y_points[1],
    )

    prob_lims = [1e-4, 1]

    logger.info(f"Plotting dataframe")

    total_count = df.count()

    emp_cdf = list()
    for y in y_points:
        delay_budget = y
        new_cond_df = df.where(df[0] <= delay_budget)
        success_count = new_cond_df.count()
        emp_success_prob = success_count / total_count
        emp_cdf.append(emp_success_prob)

    ax = tail_ax
    ax.plot(
        y_points,
        np.float64(1.00)-np.array(emp_cdf,dtype=np.float64),
        label=f"{parquet_labels[idx]}",
        color=parquet_color_pallete[idx],
        linestyle=parquet_line,
    )

    if plot_model:
        logger.info(f"Plotting model")

        logger.info("loading a non-conditional model")
        if model_info["modelname"] == "gmm":
            model = GaussianMM(h5_addr=model_h5_addr)
        elif model_info["modelname"] == "gmevm":
            model = GaussianMixtureEVM(h5_addr=model_h5_addr)
        else:
            raise Exception("wrong model")

        # create x_train and y_train
        Y = np.array(y_points).astype(np.float64) - (np.float64(target_mean)*np.float64(target_scale))

        logger.info(f"shape of target Y: {Y.shape}")
        prob, logprob, pred_cdf = model.prob_batch(Y)

        ax.plot(
            y_points,
            np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
            label=f"{idx}",
        )

ax.set_yscale('log')
ax.set_ylim(prob_lims[0],prob_lims[1])             
ax.set_xlabel("Link delay [ms]")
ax.set_ylabel("Complementary CDF")
ax.grid(True, which="both", ls="-")
#ax.legend()
ax.legend(custom_legend_lines, custom_legend_items)

tail_fig.savefig(result_file)