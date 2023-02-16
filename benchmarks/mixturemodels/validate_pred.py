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


def parse_validate_pred_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:m:l:r:c:y:",
            [
                "dataset=",
                "target=",
                "condition-nums=",
                "models=",
                "label=",
                "rows=",
                "columns=",
                "y-points="
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    args_dict["y_points"] = [0, 100, 400]
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
        elif opt in ("-x", "--condition-nums"):
            args_dict["condition_nums"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-m", "--models"):
            args_dict["models"] = [s.strip().split(".") for s in arg.split(",")]
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]

    return args_dict

def lookup_df(folder_path, cond_num, spark):
    json_path = os.path.join(folder_path, f"{cond_num}_conditions.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    parquet_path = os.path.join(folder_path, f"{cond_num}_records.parquet")
    cond_df = spark.read.parquet(parquet_path)
    total_count = cond_df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return info_json, cond_df

def run_validate_pred_processes(exp_args: list):
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

    # this project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path,cond_num,spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

    key_label = exp_args["target"]

    # figure
    nrows = exp_args["rows"]
    ncols = exp_args["columns"]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flat

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

        # plot measurements
        ax = axes[idx]
        emp_pdf = np.diff(np.array(emp_cdf))
        emp_pdf = np.append(emp_pdf,[0])*exp_args["y_points"][1]/(exp_args["y_points"][2]-exp_args["y_points"][0])
        ax.plot(
            y_points,
            emp_pdf,
            marker=".",
            label="measurements pdf",
        )

        # plot predictions
        for model_list in exp_args["models"]:
            model_project_name = model_list[0]
            model_conf_key = model_list[1]
            ensemble_num = model_list[2]
            model_path = (
                main_path + model_project_name + "_results/" + model_conf_key + "/"
            )

            with open(
                model_path + f"model_{ensemble_num}.json"
            ) as json_file:
                model_dict = json.load(json_file)

            if model_dict["type"] == "gmm":
                pr_model = ConditionalGaussianMM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )
            elif model_dict["type"] == "gmevm":
                pr_model = ConditionalGammaMixtureEVM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )

            cond_val_list = []
            for cond_label in cond_dict:
                if cond_label in model_dict["condition_labels"]:
                    if isinstance(cond_dict[cond_label],list):
                        cond_val_list.append(sum(cond_dict[cond_label]) / 2)
                    else:
                        cond_val_list.append(cond_dict[cond_label])

            x = np.repeat(
                [cond_val_list], len(y_points), axis=0
            )
            y = np.array(y_points, dtype=np.float64)
            y = y.clip(min=0.00)
            prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
            ax.plot(
                y_points,
                prob,
                marker=".",
                label="prediction " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            )


        ax.set_title(f"{cond_dict}")
        ax.set_xlabel(key_label)
        ax.set_ylabel("probability")
        ax.grid()
        ax.legend()

    # pdf figure
    fig.tight_layout()
    fig.savefig(project_path + "validate_pred.png")