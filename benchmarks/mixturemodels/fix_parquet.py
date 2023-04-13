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
import random
from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")

# This script opens all parquet files in a folder and change the column types 

spark = (
    SparkSession.builder.master("local")
    .appName("LoadParquets")
    .config("spark.executor.memory", "6g")
    .config("spark.driver.memory", "70g")
    .config("spark.driver.maxResultSize", 0)
    .getOrCreate()
)

# folder setting
p = Path(__file__).parents[0]
main_path = str(p) + "/"

# dataset project folder setting
dataset_project_path = main_path + "ep5g/measurement_training_results/"

# open the empirical dataset
all_files = os.listdir(dataset_project_path)

for f in all_files:
    if f.endswith(".parquet"):
        df = spark.read.parquet(dataset_project_path+f)
        total_count = df.count()
        logger.info(f"Parquet file {f} is loaded.")
        logger.info(f"Total number of samples in this file: {total_count}")
        
        # convert columns X and Y to double
        df = df.withColumn("X",df.X.cast('double'))
        df = df.withColumn("Y",df.Y.cast('double'))

        df.printSchema()

        pddf = df.toPandas()
        pddf.to_parquet(
            dataset_project_path + 'f_' + f,
        )