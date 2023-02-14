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
            "hq:w:d:m:l:r:c:y:e:",
            [
                "qlens=",
                "ldps=",
                "dataset=",
                "models=",
                "label=",
                "rows=",
                "columns=",
                "y-points=",
                "ensemble=",
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
        elif opt in ("-q", "--qlens"):
            s = "[" + arg + "]"
            args_dict["qlens"] = ast.literal_eval(s.strip())
        elif opt in ("-w", "--ldps"):
            s = "[" + arg + "]"
            args_dict["ldps"] = ast.literal_eval(s.strip())
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
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
        elif opt in ("-e", "--ensemble-num"):
            args_dict["ensemble_num"] = int(arg)

    return args_dict


def lookup_run_number(folder_path, qlens, ldps):
    results = []
    # Get a list of all files in the directory
    all_files = os.listdir(folder_path)
    # Filter the list to only include files that match the pattern "*_info.json"
    for file_name in all_files:
        if file_name.endswith("_info.json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                info_json = json.load(file)
                if info_json["qlens"] == qlens and info_json["ldps"] == ldps:
                    results.append(int(file_name.split("_")[0]))

    if results == []:
        logger.error(f"No run with qlens {qlens} and ldps {ldps} found")
    return results


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

    # find runs with the desired qlen and ldp
    # inputs: exp_args["qlens"] and exp_args["ldps"]
    conditions = []
    for ldps in exp_args["ldps"]:
        for qlens in exp_args["qlens"]:
            run_nums = lookup_run_number(dataset_project_path, qlens, ldps)
            conditions.append(
                {
                    "qlens": qlens,
                    "ldps": ldps,
                    "run_nums": run_nums,
                }
            )

    # condition_labels = ["queue_length","ldp"]
    key_label = "end2end_delay"

    # figure 1
    nrows = exp_args["rows"]
    ncols = exp_args["columns"]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flat

    for idx, records_dict in enumerate(conditions):
        ax = axes[idx]

        # open the empirical dataset
        all_files = os.listdir(dataset_project_path)
        files = []
        for f in all_files:
            if f.endswith(".parquet"):
                for run_num in records_dict["run_nums"]:
                    if f.startswith(f"{run_num}_"):
                        files.append(dataset_project_path + "/" + f)

        cond_df = spark.read.parquet(*files)
        total_count = cond_df.count()
        logger.info(f"Parquet files {files} are loaded.")
        logger.info(f"Total number of samples in this empirical dataset: {total_count}")

        emp_cdf = list()
        for y in y_points:
            delay_budget = y
            new_cond_df = cond_df.where(cond_df[key_label] <= delay_budget)
            success_count = new_cond_df.count()
            emp_success_prob = success_count / total_count
            emp_cdf.append(emp_success_prob)

        ax.plot(
            y_points,
            emp_cdf,
            marker=".",
            label="simulation",
        )

        for model_list in exp_args["models"]:
            model_project_name = model_list[0]
            model_conf_key = model_list[1]
            model_path = (
                main_path + model_project_name + "_results/" + model_conf_key + "/"
            )

            with open(
                model_path + f"model_{exp_args['ensemble_num']}.json"
            ) as json_file:
                model_dict = json.load(json_file)

            if model_dict["type"] == "gmm":
                pr_model = ConditionalGaussianMM(
                    h5_addr=model_path + f"model_{exp_args['ensemble_num']}.h5",
                )
            elif model_dict["type"] == "gmevm":
                pr_model = ConditionalGammaMixtureEVM(
                    h5_addr=model_path + f"model_{exp_args['ensemble_num']}.h5",
                )

            x = np.repeat(
                [[*records_dict["qlens"], *records_dict["ldps"]]], len(y_points), axis=0
            )
            y = np.array(y_points, dtype=np.float64)
            y = y.clip(min=0.00)
            prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
            ax.plot(
                y_points,
                pred_cdf,
                marker=".",
                label="prediction " + model_project_name + "." + model_conf_key,
            )

        ax.set_title(f"qlen={records_dict['qlens']},ldps={records_dict['ldps']}")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Success probability")
        ax.grid()
        ax.legend()
    # figure
    fig.tight_layout()
    fig.savefig(project_path + f"qlen_validation_bulk_{exp_args['ensemble_num']}.png")
