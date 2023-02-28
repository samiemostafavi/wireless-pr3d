import getopt
import json
import os
import sys
from pathlib import Path
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


def parse_plot_evaluation_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hp:m:x:y:t:",
            ["project=", "models=", "condition-nums=", "y-points=", "type="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m predictors_benchmark -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m delay_bound_benchmark plot "
                + "-a <arrival rate> -u <until> -l <label>",
            )
            sys.exit()
        elif opt in ("-p", "--project"):
            # project folder setting
            p = Path(__file__).parents[0]
            args_dict["project_folder"] = str(p) + "/" + arg + "_results/"
        elif opt in ("-m", "--models"):
            args_dict["models"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-x", "--condition-nums"):
            args_dict["condition_nums"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-t", "--type"):
            args_dict["type"] = arg

    return args_dict


def plot_evaluation_main(exp_args):

    logger.info(f"Plotting benchmark results with args: {exp_args}")

    # set project folder
    project_folder = exp_args["project_folder"]

    plt.style.use(["science", "ieee", "bright"])

    # read csvs inside
    for idx,cond_num in enumerate(exp_args["condition_nums"]):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        logger.info(f"Plotting conditional results {cond_num}")
        df = pd.read_csv(os.path.join(project_folder, f"{cond_num}.csv"))
        y_points = df["y"].to_numpy()

        # calc error
        for column in df:
            if column == "y" or column == "tail.measurements":
                continue
            if column.split(".")[1] in exp_args["models"]:
                df[column] = df.apply(lambda x: abs(x["tail.measurements"] - x[column]), axis=1)
            else:
                df.drop([column],axis=1,inplace=True)

        # take min, max, avg
        for model in exp_args["models"]:
            # find columns
            model_cols = []
            for column in df:
                if column == "y" or column == "tail.measurements":
                    continue
                if column.split(".")[1] == model:
                    model_cols.append(column)

            df[f"{model}.max"] = df[model_cols].quantile(0.95,axis=1)
            df[f"{model}.min"] = df[model_cols].quantile(0.05,axis=1)
            #df[f"{model}.max"] = df[model_cols].max(axis=1)
            #df[f"{model}.min"] = df[model_cols].min(axis=1)
            df[f"{model}.mean"] = df[model_cols].mean(axis=1)

            ax.plot(y_points, df[f"{model}.mean"], '-', label=model)
            ax.fill_between(y_points, df[f"{model}.min"].to_numpy(), df[f"{model}.max"].to_numpy(), alpha=0.2)

            ax.set_xlabel("Delay [ms]")
            ax.set_ylabel("Error [log]")
            ax.grid()
            ax.legend()

        # cdf figure
        fig.tight_layout()
        fig.savefig(project_folder + f"{cond_num}.png")