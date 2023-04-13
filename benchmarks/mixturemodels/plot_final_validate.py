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
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pr3d.de import ConditionalGaussianMixtureEVM, ConditionalGaussianMM
from pyspark.sql import SparkSession

import scienceplots
plt.style.use(['science','ieee'])

warnings.filterwarnings("ignore")

markersize=5
darker = 0.9
MeasColorPalette = [
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1],
    [0.1*darker,0.2*darker,0.8*darker, 1]
]
lighter = 1
PredColorPalette = [
    [
        [0.9*lighter,0.4*lighter,0.0*lighter, 1],
        [0.9*lighter,0.4*lighter,0.0*lighter, 1],
        [0.9*lighter,0.4*lighter,0.0*lighter, 1]
    ],
    [
        [0.2*lighter,0.8*lighter,0.1*lighter, 1],
        [0.2*lighter,0.8*lighter,0.1*lighter, 1],
        [0.2*lighter,0.8*lighter,0.1*lighter, 1]
    ]
]
PredColorPalette = np.array(PredColorPalette)
np.clip(PredColorPalette, 0, 1, out=PredColorPalette)
PredLineWidth=1
PredLine='-'

xaxis_label = 'Link delay [ms]'

meas_legend_labels = [
    'meas. MCS=3',
    'meas. MCS=5',
    'meas. MCS=7'
]

pred_legend_labels = [
    [
        '_hidden',
        '_hidden',
        'pred. GMEVM'
    ],
    [
        '_hidden',
        '_hidden',
        'pred. GMM'
    ],
]

markerevery=(0.03,0.03)

def parse_plot_final_validate_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:n:m:l:r:c:y:u:f:i:p:g:o:",
            [
                "dataset=",
                "target=",
                "condition-nums=",
                "condition-markers=",
                "models=",
                "label=",
                "rows=",
                "columns=",
                "y-points=",
                "prob-lims=",
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
    args_dict["prob_lims"] = None
    args_dict["condition_markers"] = None
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
        elif opt in ("-n", "--condition-markers"):
            args_dict["condition_markers"] = [s.strip() for s in arg.split(",")]
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
        elif opt in ("-u", "--prob-lims"):
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


    if not args_dict["condition_markers"]:
        args_dict["condition_markers"] = ["." for cond in args_dict["condition_nums"]]

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

def run_plot_final_validate_processes(exp_args: list):
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

    # bulk plot axis
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

    single_plot = False
    if exp_args["rows"] == 0 and exp_args["columns"] == 0:
        single_plot = True

    # figure
    # CDF figure
    if exp_args["plotcdf"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            cdf_fig, cdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            cdf_axes = cdf_axes.flat
        else:
            cdf_fig, cdf_ax = plt.subplots(nrows=1, ncols=1)

    # Tail figure
    if exp_args["plottail"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            tail_fig, tail_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            tail_axes = tail_axes.flat
        else:
            tail_fig, tail_ax = plt.subplots(nrows=1, ncols=1)

    # PDF figure
    if exp_args["plotpdf"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            pdf_fig, pdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            pdf_axes = pdf_axes.flat
        else:
            pdf_fig, pdf_ax = plt.subplots(nrows=1, ncols=1)


    for idx, cond_dict in enumerate(conditions):
        logger.info(f"Plotting dataframe {idx} with conditions {cond_dict}")
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

        # Calculate emp PDF
        emp_pdf = np.diff(np.array(emp_cdf))
        emp_pdf = np.append(emp_pdf,[0])*exp_args["y_points"][1]/(exp_args["y_points"][2]-exp_args["y_points"][0])

        if exp_args["plotcdf"]:
            if not single_plot:
                ax = cdf_axes[idx]
            else:
                ax = cdf_ax
            ax.plot(
                y_points,
                emp_cdf,
                marker=exp_args["condition_markers"][idx],
                markersize=markersize,
                markevery=markerevery,
                color=MeasColorPalette[idx],
                label=meas_legend_labels[idx]
            )

        if exp_args["plottail"]:
            if not single_plot:
                ax = tail_axes[idx]
            else:
                ax = tail_ax
            ax.plot(
                y_points,
                np.float64(1.00)-np.array(emp_cdf,dtype=np.float64),
                marker=exp_args["condition_markers"][idx],
                markersize=markersize,
                markevery=markerevery,
                color=MeasColorPalette[idx],
                label=meas_legend_labels[idx]
            )

        if exp_args["plotpdf"]:
            if not single_plot:
                ax = pdf_axes[idx]
            else:
                ax = pdf_ax
            ax.plot(
                y_points,
                emp_pdf,
                marker=exp_args["condition_markers"][idx],
                markersize=markersize,
                markevery=markerevery,
                color=MeasColorPalette[idx],
                label=meas_legend_labels[idx]
            )

        # plot predictions
        for idy,model_list in enumerate(exp_args["models"]):
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
                pr_model = ConditionalGaussianMixtureEVM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )

            # define x from the conditional dataset
            # select the columns
            cond_columns = []
            for cond_label in model_dict["condition_labels"]:
                cond_columns.append(cond_label)

            # select columns and sample the rows
            rows = cond_df.select(cond_columns).sample(False, (len(y_points_standard)*2)/cond_df.count(), seed=0).limit(len(y_points_standard))
            x_rows = rows.collect()

            # define x numpy list
            x_list = []
            for row in x_rows:
                x_list.append(
                    [ row[colm] for colm in cond_columns ]
                )
            x = np.array(x_list)

            # define y points and run the inference
            y = np.array(y_points_standard, dtype=np.float64)
            #y = y.clip(min=0.00)
            prob, logprob, pred_cdf = pr_model.prob_batch(x, y)

            if exp_args["plotcdf"]:
                if not single_plot:
                    ax = cdf_axes[idx]
                else:
                    ax = cdf_ax
                ax.plot(
                    y_points,
                    pred_cdf,
                    marker="",
                    color=PredColorPalette[idy][idx],
                    linewidth=PredLineWidth,
                    linestyle=PredLine,
                    label=pred_legend_labels[idy][idx]
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                ax.set_xlabel(xaxis_label)
                ax.set_ylabel("Success probability")
                ax.grid(visible=True)
                ax.legend()

            if exp_args["plottail"]:
                if not single_plot:
                    ax = tail_axes[idx]
                else:
                    ax = tail_ax
                ax.plot(
                    y_points,
                    np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
                    marker="",
                    color=PredColorPalette[idy][idx],
                    linewidth=PredLineWidth,
                    linestyle=PredLine,
                    label=pred_legend_labels[idy][idx]
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])
                    
                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                ax.set_xlabel(xaxis_label)
                ax.set_ylabel("Tail probability")
                ax.grid(visible=True)
                ax.legend()

            if exp_args["plotpdf"]:
                if not single_plot:
                    ax = pdf_axes[idx]
                else:
                    ax = pdf_ax
                ax.plot(
                    y_points,
                    prob,
                    marker="",
                    color=PredColorPalette[idy][idx],
                    linewidth=PredLineWidth,
                    linestyle=PredLine,
                    label=pred_legend_labels[idy][idx]
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                ax.set_xlabel(xaxis_label)
                ax.set_ylabel("probability")
                ax.grid(visible=True)
                ax.legend()


    if exp_args["plotcdf"]:
        # cdf figure
        cdf_fig.tight_layout()
        if exp_args["logplot"]:
            cdf_fig.savefig(project_path + f"{key_label}_log_cdf.png")
            pickle.dump(cdf_fig,open(project_path + f"{key_label}_log_cdf.pickle",'wb'))
        elif exp_args["loglogplot"]:
            cdf_fig.savefig(project_path + f"{key_label}_loglog_cdf.png")
            pickle.dump(cdf_fig,open(project_path + f"{key_label}_loglog_cdf.pickle",'wb'))
        else:
            cdf_fig.savefig(project_path + f"{key_label}_cdf.png")
            pickle.dump(cdf_fig,open(project_path + f"{key_label}_cdf.pickle",'wb'))

    if exp_args["plottail"]:
        # cdf figure
        tail_fig.tight_layout()
        if exp_args["logplot"]:
            tail_fig.savefig(project_path + f"{key_label}_log_tail.png")
            pickle.dump(tail_fig,open(project_path + f"{key_label}_log_tail.pickle",'wb'))
        elif exp_args["loglogplot"]:
            tail_fig.savefig(project_path + f"{key_label}_loglog_tail.png")
            pickle.dump(tail_fig,open(project_path + f"{key_label}_loglog_tail.pickle",'wb'))
        else:
            tail_fig.savefig(project_path + f"{key_label}_tail.png")
            pickle.dump(tail_fig,open(project_path + f"{key_label}_tail.pickle",'wb'))

    if exp_args["plotpdf"]:
        # pdf figure
        pdf_fig.tight_layout()
        if exp_args["logplot"]:
            pdf_fig.savefig(project_path + f"{key_label}_log_pdf.png")
            pickle.dump(pdf_fig,open(project_path + f"{key_label}_log_pdf.pickle",'wb'))
        elif exp_args["loglogplot"]:
            pdf_fig.savefig(project_path + f"{key_label}_loglog_pdf.png")
            pickle.dump(pdf_fig,open(project_path + f"{key_label}_loglog_pdf.pickle",'wb'))
        else:
            pdf_fig.savefig(project_path + f"{key_label}_pdf.png")
            pickle.dump(pdf_fig,open(project_path + f"{key_label}_pdf.pickle",'wb'))