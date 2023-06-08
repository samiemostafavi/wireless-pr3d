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
from pyspark.sql.functions import rand

warnings.filterwarnings("ignore")


def parse_prep_dataset_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:l:r:c:y:n:s:f:p:g:w",
            ["dataset=", "target=", "conditions=", "label=", "rows=", "columns=", "y-points=", "normalize=", "size=", "plot-pdf", "plot-cdf", "log","preview"],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    # default values
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["logplot"] = False
    args_dict["preview"] = False
    args_dict["normalize"] = None
    args_dict["cond_ds_size"] = None
    args_dict["conditions"] = None

    # parse the args
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark validate "
                + "-q <qlens> -d <dataset> -m <trained models> -l <label> -e <ensemble num>",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-t", "--target"):
            args_dict["target"] = arg
        elif opt in ("-x", "--conditions"):
            args_dict["conditions"] = json.loads(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-n", "--normalize"):
            args_dict["normalize"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-s", "--size"):
            args_dict["cond_ds_size"] = int(arg)
        elif opt in ("-f", "--plot-cdf"):
            args_dict["plotcdf"] = True
        elif opt in ("-p", "--plot-pdf"):
            args_dict["plotpdf"] = True
        elif opt in ("-g", "--log"):
            args_dict["logplot"] = True
        elif opt in ("-w", "--preview"):
            args_dict["preview"] = True

    return args_dict


def run_prep_dataset_processes(exp_args: list):
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

    # bulk plot axis
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )

    # folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"

    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"


    # open the empirical dataset
    all_files = os.listdir(dataset_project_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(dataset_project_path + f)

    df = spark.read.parquet(*files)
    total_count = df.count()
    logger.info(f"Parquet files {files} are loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    #Get All column names and it's types
    logger.info("Dataset preview:")
    df.printSchema()
    
    if exp_args["preview"]:
        df.summary().show()
        return

    #if exp_args["scale"]:
    from pyspark.sql.functions import col, mean, randn

    noise_seed = random.randint(0,1000)
    send_mean = df.select(mean('send')).collect()[0][0]
    send_scale = 1e-6
    send_noise_variance = 1
    send_noise_highvariance = 3
    df = df.withColumn('send_scaled', (col('send') * send_scale))
    df = df.withColumn('send_standard', ((col('send')-send_mean) * send_scale))
    df = df.withColumn('send_noisy', col('send_standard') + (randn(seed=noise_seed)*send_noise_variance))
    df = df.withColumn('send_verynoisy', col('send_standard') + (randn(seed=noise_seed)*send_noise_highvariance))

    receive_mean = df.select(mean('receive')).collect()[0][0]
    receive_scale = 1e-6
    receive_noise_variance = 1
    receive_noise_highvariance = 3
    df = df.withColumn('receive_scaled', (col('receive') * receive_scale))
    df = df.withColumn('receive_standard', ((col('receive')-receive_mean) * receive_scale))
    df = df.withColumn('receive_noisy', col('receive_standard') + (randn(seed=noise_seed)*receive_noise_variance))
    df = df.withColumn('receive_verynoisy', col('receive_standard') + (randn(seed=noise_seed)*receive_noise_highvariance))

    rtt_mean = df.select(mean('rtt')).collect()[0][0]
    rtt_scale = 1e-6
    rtt_noise_variance = 1
    rtt_noise_highvariance = 3
    df = df.withColumn('rtt_scaled', (col('rtt') * rtt_scale))
    df = df.withColumn('rtt_standard', ((col('rtt')-rtt_mean) * rtt_scale))
    df = df.withColumn('rtt_noisy', col('rtt_standard') + (randn(seed=noise_seed)*rtt_noise_variance))
    df = df.withColumn('rtt_verynoisy', col('rtt_standard') + (randn(seed=noise_seed)*rtt_noise_highvariance))


    means_dict = {
        "send":{
            "mean":send_mean,
            "scale":send_scale
        },
        "receive":{
            "mean":receive_mean,
            "scale":receive_scale
        },
        "rtt":{
            "mean":rtt_mean,
            "scale":rtt_scale
        }
    }

    if exp_args["normalize"]:
        from pyspark.ml.feature import MinMaxScaler
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml import Pipeline
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        from pyspark.sql.functions import min, max

        # UDF for converting column type from vector to double type
        unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

        # Iterating over columns to be scaled
        for i in exp_args["normalize"]:
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

        logger.info("Dataset after normalization:")
        df.summary().show()

    logger.info("Dataset preview after scales and normalizations:")
    #df.summary().show()

    # this project folder setting
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    # save json file
    with open(
        project_path + f"info.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(means_dict, indent = 4))

    if exp_args["conditions"] is None:
        # append the conditional df to the list
        logger.info(f"Dataframe with no conditions has {df.count()} samples.")

        if exp_args["cond_ds_size"]:
            # shuffle samples
            df = df.orderBy(rand())
            # take the desired number of records
            df = df.sample(
                withReplacement=False,
                fraction=exp_args["cond_ds_size"] / df.count(),
                seed=12345,
            )

        logger.info(f"Writing {df.count()} samples to the parquet file.")

        # save the parquet file
        pd_df = df.toPandas()
        pd_df.to_parquet(
            project_path + "0_records.parquet",
        )
        return
    
    # create conditional dataframes
    empt_dict = { "cond_dataset_num" : None }
    dim_tuple = ()
    for cond_label in exp_args["conditions"]:
        empt_dict = { **empt_dict, cond_label : None }
        dim_tuple = dim_tuple + (len(exp_args["conditions"][cond_label]),)

    logger.info(f"Chosen conditions dimensions: {dim_tuple}")
      
    dim_list = list(dim_tuple)
    cond_dataframes = []
    conditions = []
    for idx in itertools.product(*[range(s) for s in dim_list]):
        # idx would be (0,0,1) or (1,2,3) in case of having 3 conditions
        # i would be "idx[0]", j "idx[1]" and so on...
        condition_dict = {}

        # copy the original df
        cond_df = df.alias(f"cond_df_{len(cond_dataframes)}")
        
        for jdx, cond_label in enumerate(exp_args["conditions"]):
            cond = exp_args["conditions"][cond_label][idx[jdx]]
            condition_dict[cond_label] = cond

            if isinstance(cond, list):
                cond_df = cond_df.filter(
                    cond_df[cond_label].between(cond[0], cond[1]),
                )
            else:
                cond_df = cond_df.filter(
                    df[cond_label] == cond,
                )

        if cond_df.count():
            # append the conditional df to the list
            logger.info(f"Dataframe {len(cond_dataframes)} for conditions {condition_dict} has {cond_df.count()} samples.")
            #cond_df.summary().show()

            # save json file
            with open(
                project_path + f"{len(cond_dataframes)}_conditions.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(condition_dict, indent = 4))


            if exp_args["cond_ds_size"]:
                # shuffle samples
                cond_df = cond_df.orderBy(rand())
                # take the desired number of records
                cond_df = cond_df.sample(
                    withReplacement=False,
                    fraction=exp_args["cond_ds_size"] / cond_df.count(),
                    seed=12345,
                )

            logger.info(f"Writing {cond_df.count()} samples to the parquet file.")

            # save the parquet file
            pd_cond_df = cond_df.toPandas()
            pd_cond_df.to_parquet(
                project_path + f"{len(cond_dataframes)}_records.parquet",
            )

            # append the conditions dict and dataframe to the lists
            cond_dataframes.append(cond_df)
            conditions.append(condition_dict)
        else:
            logger.info(f"No samples were found for conditions {condition_dict}.")
