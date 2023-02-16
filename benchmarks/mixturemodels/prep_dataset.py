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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")


def parse_prep_dataset_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:l:r:c:y:f:p:w",
            ["dataset=", "target=", "conditions=", "label=", "rows=", "columns=", "y-points=", "plot-pdf", "plot-cdf","preview"],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    # default values
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["preview"] = False

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
        elif opt in ("-f", "--plot-cdf"):
            args_dict["plotcdf"] = True
        elif opt in ("-p", "--plot-pdf"):
            args_dict["plotpdf"] = True
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
    df.summary().show()

    if exp_args["preview"]:
        return

    # this project folder setting
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

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
            logger.info(f"Dataframe {len(cond_dataframes)} for conditions {condition_dict} has {cond_df.count()} samples with the following preview:")
            cond_df.summary().show()
        
            # save json file
            with open(
                project_path + f"{len(cond_dataframes)}_conditions.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(condition_dict, indent = 4))

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
            

    # CDF figure
    if exp_args["plotcdf"]:
        nrows = exp_args["rows"]
        ncols = exp_args["columns"]
        cdf_fig, cdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
        cdf_axes = cdf_axes.flat

    # PDF figure
    if exp_args["plotpdf"]:
        nrows = exp_args["rows"]
        ncols = exp_args["columns"]
        pdf_fig, pdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
        pdf_axes = pdf_axes.flat

    if exp_args["plotpdf"] or exp_args["plotcdf"]:
        key_label = exp_args["target"]
        for idx, cond_dict in enumerate(conditions):
            logger.info(f"Plotting dataframe {idx} with conditions {cond_dict}")
            cond_df = cond_dataframes[idx]
            total_count = cond_df.count()
            emp_cdf = list()
            for y in y_points:
                delay_budget = y
                new_cond_df = cond_df.where(cond_df[key_label] <= delay_budget)
                success_count = new_cond_df.count()
                emp_success_prob = success_count / total_count
                emp_cdf.append(emp_success_prob)

            if exp_args["plotcdf"]:
                ax = cdf_axes[idx]
                ax.plot(
                    y_points,
                    emp_cdf,
                    marker=".",
                    label="cdf",
                )

                ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("Success probability")
                ax.grid()
                ax.legend()

            if exp_args["plotpdf"]:
                ax = pdf_axes[idx]
                emp_pdf = np.diff(np.array(emp_cdf))
                emp_pdf = np.append(emp_pdf,[0])
                ax.plot(
                    y_points,
                    emp_pdf,
                    marker=".",
                    label="pdf",
                )

                ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("probability")
                ax.grid()
                ax.legend()

        if exp_args["plotcdf"]:
            # cdf figure
            cdf_fig.tight_layout()
            cdf_fig.savefig(project_path + "prepped_dataset_cdf.png")

        if exp_args["plotpdf"]:
            # pdf figure
            pdf_fig.tight_layout()
            pdf_fig.savefig(project_path + "prepped_dataset_pdf.png")
