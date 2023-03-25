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


def parse_plot_prepped_dataset_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:r:c:y:l:f:i:p:g:o:w",
            ["dataset=", "target=", "condition-nums=", "rows=", "columns=", "y-points=", "prob-lims=", "plot-pdf", "plot-cdf", "plot-tail", "log", "loglog", "preview"],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    # default values
    args_dict["prob_lims"] = None
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["plottail"] = False
    args_dict["logplot"] = False
    args_dict["loglogplot"] = False
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
        elif opt in ("-x", "--condition-nums"):
            args_dict["condition_nums"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-l", "--log-lims"):
            args_dict["prob_lims"] = [np.float64(s.strip()) for s in arg.split(",")]
        elif opt in ("-f", "--plot-cdf"):
            args_dict["plotcdf"] = True
        elif opt in ("-i", "--plot-tail"):
            args_dict["plottail"] = True
        elif opt in ("-p", "--plot-pdf"):
            args_dict["plotpdf"] = True
        elif opt in ("-g", "--log"):
            args_dict["logplot"] = True
        elif opt in ("-o", "--loglog"):
            args_dict["loglogplot"] = True
        elif opt in ("-w", "--preview"):
            args_dict["preview"] = True

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

def run_plot_prepped_dataset_processes(exp_args: list):
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

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path,cond_num,spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

    key_label = exp_args["target"]

    # CDF figure
    if exp_args["plotcdf"]:
        nrows = exp_args["rows"]
        ncols = exp_args["columns"]
        cdf_fig, cdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
        cdf_axes = cdf_axes.flat

    # Tail figure
    if exp_args["plottail"]:
        nrows = exp_args["rows"]
        ncols = exp_args["columns"]
        tail_fig, tail_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
        tail_axes = tail_axes.flat

    # PDF figure
    if exp_args["plotpdf"]:
        nrows = exp_args["rows"]
        ncols = exp_args["columns"]
        pdf_fig, pdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
        pdf_axes = pdf_axes.flat

    if exp_args["plotpdf"] or exp_args["plotcdf"] or exp_args["plottail"]:
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
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("Success probability")
                ax.grid()
                ax.legend()

            if exp_args["plottail"]:
                ax = tail_axes[idx]
                ax.plot(
                    y_points,
                    np.float64(1.00)-np.array(emp_cdf,dtype=np.float64),
                    marker=".",
                    label="tail",
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])
                    
                ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("Fail probability")
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
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                
                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("probability")
                ax.grid()
                ax.legend()

        if exp_args["plotcdf"]:
            # cdf figure
            cdf_fig.tight_layout()
            if exp_args["logplot"]:
                cdf_fig.savefig(dataset_project_path + f"{key_label}_log_cdf.png")
            elif exp_args["loglogplot"]:
                cdf_fig.savefig(dataset_project_path + f"{key_label}_loglog_cdf.png")
            else:
                cdf_fig.savefig(dataset_project_path + f"{key_label}_cdf.png")

        if exp_args["plottail"]:
            # cdf figure
            tail_fig.tight_layout()
            if exp_args["logplot"]:
                tail_fig.savefig(dataset_project_path + f"{key_label}_log_tail.png")
            elif exp_args["loglogplot"]:
                tail_fig.savefig(dataset_project_path + f"{key_label}_loglog_tail.png")
            else:
                tail_fig.savefig(dataset_project_path + f"{key_label}_tail.png")

        if exp_args["plotpdf"]:
            # pdf figure
            pdf_fig.tight_layout()
            if exp_args["logplot"]:
                pdf_fig.savefig(dataset_project_path + f"{key_label}_log_pdf.png")
            elif exp_args["loglogplot"]:
                pdf_fig.savefig(dataset_project_path + f"{key_label}_loglog_pdf.png")
            else:
                pdf_fig.savefig(dataset_project_path + f"{key_label}_pdf.png")
