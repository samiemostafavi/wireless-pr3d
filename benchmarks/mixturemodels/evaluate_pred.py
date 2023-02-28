import ast
import getopt
import json
import multiprocessing as mp
import os
import sys
import time
import re
import warnings
from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pr3d.de import ConditionalGaussianMixtureEVM, ConditionalGaussianMM
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")


def parse_evaluate_pred_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:m:l:r:c:y:f:i:p:g:o:",
            [
                "dataset=",
                "target=",
                "condition-nums=",
                "models=",
                "label=",
                "rows=",
                "columns=",
                "y-points=",
                "plot-pdf", 
                "plot-cdf", 
                "plot-tail", 
                "log",
                "loglog",
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    # default values
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["plottail"] = False
    args_dict["logplot"] = False
    args_dict["loglogplot"] = False

    # parse the args
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark validate "
                + "-q <qlens> -d <dataset> -m <trained models> -l <label> -e <ensemble num> ",
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

def run_evaluate_pred_processes(exp_args: list):
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
    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path,cond_num,spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)


    json_path = os.path.join(dataset_project_path, "info.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    key_label = exp_args["target"]
    key_mean = info_json[key_label]["mean"]
    key_scale = info_json[key_label]["scale"]

    # evaluation axis
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )
    y_points_standard = np.linspace(
        start=exp_args["y_points"][0]-(key_mean*key_scale),
        stop=exp_args["y_points"][2]-(key_mean*key_scale),
        num=exp_args["y_points"][1],
    )

    for idx, cond_dict in enumerate(conditions):
        logger.info(f"Evaluating dataframe {idx} with conditions {cond_dict}")
        cond_df = cond_dataframes[idx]

        # Calculate emp CDF
        total_count = cond_df.count()
        emp_cdf = list()
        for y in y_points:
            delay_budget = y
            new_cond_df = cond_df.where(cond_df[key_label+'_scaled'] <= delay_budget)
            success_count = new_cond_df.count()
            emp_success_prob = success_count / total_count
            emp_cdf.append(emp_success_prob)

        emp_tail = np.log10(np.float64(1.00)-np.array(emp_cdf,dtype=np.float64))
        
        res_df = pd.DataFrame()
        res_df["y"] = y_points
        res_df["tail.measurements"] = emp_tail

        # predictions
        for model_list in exp_args["models"]:
            
            model_project_name = model_list[0]
            model_conf_key = model_list[1]
            model_path = (
                main_path + model_project_name + "_results/" + model_conf_key + "/"
            )

            ensemble_nums = []
            for file in os.listdir(model_path):
                if file.startswith("model_") and file.endswith(".json"):
                    number = re.search(r"\d+", file).group()
                    ensemble_nums.append(number)

            for ensemble_num in ensemble_nums:

                with open(
                    model_path + f"model_{ensemble_num}.json"
                ) as json_file:
                    model_dict = json.load(json_file)

                if model_dict["type"] == "gmm":
                    pr_model = ConditionalGaussianMM(
                        h5_addr=model_path + f"model_{ensemble_num}.h5",
                    )
                elif model_dict["type"] == "gmevm":
                    pr_model = ConditionalGaussianMixtureEVM(
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
                    [cond_val_list], len(y_points_standard), axis=0
                )
                y = np.array(y_points_standard, dtype=np.float64)
                #y = y.clip(min=0.00)
                prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
                pred_tail = np.log10(np.float64(1.00)-np.array(pred_cdf,dtype=np.float64))

                res_df[f"tail.{model_conf_key}.{ensemble_num}"]=pred_tail
            
        res_df.to_csv(project_path+f"{idx}.csv",index=False)