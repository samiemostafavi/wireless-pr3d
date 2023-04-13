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



# oai5g evaluation plot format
condition_markers = [
    '.',
    '*',
    '2'
]
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
"""

# ep5g evaluation plot format
condition_markers = [
    '.',
    '.',
    '.'
]
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
        [0.1*lighter,0.6*lighter,0.9*lighter, 1],
        [0.1*lighter,0.6*lighter,0.9*lighter, 1],
        [0.1*lighter,0.6*lighter,0.9*lighter, 1]
    ],
    [
        [0.8*lighter,0.7*lighter,0.0*lighter, 1],
        [0.8*lighter,0.7*lighter,0.0*lighter, 1],
        [0.8*lighter,0.7*lighter,0.0*lighter, 1]
    ],
    [
        [0.2*lighter,0.8*lighter,0.1*lighter, 1],
        [0.2*lighter,0.8*lighter,0.1*lighter, 1],
        [0.2*lighter,0.8*lighter,0.1*lighter, 1]
    ],
    [
        [1*lighter,0.1*lighter,0.1*lighter, 1],
        [1*lighter,0.1*lighter,0.1*lighter, 1],
        [1*lighter,0.1*lighter,0.1*lighter, 1]
    ]
]
PredColorPalette = np.array(PredColorPalette)
np.clip(PredColorPalette, 0, 1, out=PredColorPalette)
PredLineWidth=1
PredLine='-'
xaxis_label = 'Link delay [ms]'
meas_legend_labels = [
    'meas. X=0.5, Y=0.0',
    'meas. X=2.0, Y=2.5',
    'meas. X=4.0, Y=2.5'
]
pred_legend_labels = [
    [
        'pred. GMEVM',
        'pred. GMEVM',
        'pred. GMEVM'
    ],
    [
        'pred. GMM',
        'pred. GMM',
        'pred. GMM'
    ],
    [
        'pred. GMEVM comp.',
        'pred. GMEVM comp.',
        'pred. GMEVM comp.'
    ],
    [
        'pred. GMM comp.',
        'pred. GMM comp.',
        'pred. GMM comp.'
    ]
]
markerevery=(0.03,0.03)

"""


def parse_plot_evaluation_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hp:m:x:y:t:u:s:",
            ["project=", "models=", "condition-nums=", "y-points=", "type=", "prob-lims=", "single"],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m predictors_benchmark -h" for help')
        sys.exit(2)

    args_dict["prob_lims"] = None
    args_dict["single_plot"] = False    

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
        elif opt in ("-u", "--prob-lims"):
            args_dict["prob_lims"] = [np.float64(s.strip()) for s in arg.split(",")]
        elif opt in ("-s", "--single"):
            args_dict["single_plot"] = True

    return args_dict


def plot_evaluation_main(exp_args):

    logger.info(f"Plotting benchmark results with args: {exp_args}")

    # set project folder
    project_folder = exp_args["project_folder"]

    plt.style.use(["science", "ieee", "bright"])

    if exp_args["single_plot"]:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    # read csvs inside
    for idx,cond_num in enumerate(exp_args["condition_nums"]):
        if not exp_args["single_plot"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        logger.info(f"Plotting conditional results {cond_num}")
        df = pd.read_csv(os.path.join(project_folder, f"{cond_num}.csv"))
        y_points = df["y"].to_numpy()

        new_y_points = []
        filter_indices = []
        for i,e in enumerate(y_points):
            if e >= exp_args["y_points"][0] and e <= exp_args["y_points"][1]:
                new_y_points.append(e)
                filter_indices.append(i)

        y_points = new_y_points

        if exp_args["type"] == "error":
            # calc error
            for column in df:
                if column == "y" or column == "tail.measurements":
                    continue
                if column.split(".")[1]+'.'+column.split(".")[2] in exp_args["models"]:
                    df[column] = df.apply(lambda x: abs(x["tail.measurements"] - x[column]), axis=1)
                else:
                    df.drop([column],axis=1,inplace=True)
        elif exp_args["type"] == "tail":
            # calc error
            for column in df:
                if column == "y" or column == "tail.measurements":
                    continue
                if column.split(".")[1]+'.'+column.split(".")[2] not in exp_args["models"]:
                    df.drop([column],axis=1,inplace=True)


            meas = df["tail.measurements"].to_numpy()
            meas = meas[filter_indices]
            ax.plot(
                y_points,
                meas,
                marker=condition_markers[idx],
                markersize=markersize,
                markevery=markerevery,
                color=MeasColorPalette[idx],
                label=meas_legend_labels[idx]
            )

        # take min, max, avg
        for idy,model in enumerate(exp_args["models"]):
            # find columns
            model_cols = []
            for column in df:
                if column == "y" or column == "tail.measurements":
                    continue
                if column.split(".")[1]+'.'+column.split(".")[2] == model:
                    model_cols.append(column)

            df[f"{model}.max"] = df[model_cols].quantile(0.95,axis=1)
            df[f"{model}.min"] = df[model_cols].quantile(0.05,axis=1)
            #df[f"{model}.max"] = df[model_cols].max(axis=1)
            #df[f"{model}.min"] = df[model_cols].min(axis=1)
            df[f"{model}.mean"] = df[model_cols].mean(axis=1)


            preds = df[f"{model}.mean"].to_numpy()
            preds = preds[filter_indices]
            ax.plot(
                y_points, 
                preds, 
                linestyle=PredLine,
                color=PredColorPalette[idy][idx],
                linewidth=PredLineWidth,
                label=pred_legend_labels[idy][idx]
            )
            predsmin = df[f"{model}.min"].to_numpy()
            predsmin = predsmin[filter_indices]
            predsmax = df[f"{model}.max"].to_numpy()
            predsmax = predsmax[filter_indices]
            ax.fill_between(
                y_points, 
                predsmin, 
                predsmax,
                color=PredColorPalette[idy][idx],
                linewidth=0.0,
                alpha=0.2
            )

            ax.set_xlabel("Link delay [ms]")
            if exp_args["type"] == "error":
                ax.set_ylabel("Error [log]")
            else:
                ax.set_yscale('log')
                ax.set_ylabel("Tail probability")

            if exp_args["prob_lims"]:
                ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

            ax.grid(visible=True)
            ax.legend()

        if not exp_args["single_plot"]:
            # save figure
            fig.tight_layout()
            fig.savefig(project_folder + f"{cond_num}.png")
    
    if exp_args["single_plot"]:
        # save figure
        fig.tight_layout()
        fig.savefig(project_folder + "agg.png")