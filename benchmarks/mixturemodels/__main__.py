import getopt
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import os
import signal
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from .plot import parse_plot_args, plot_main
from .train import parse_train_args, run_train_processes
from .prep_dataset import parse_prep_dataset_args, run_prep_dataset_processes
from .validate_pred import parse_validate_pred_args, run_validate_pred_processes

# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":

    argv = sys.argv[1:]
    if argv[0] == "prep_dataset":
        validate_gym_args = parse_prep_dataset_args(argv[1:])
        run_prep_dataset_processes(validate_gym_args)
    elif argv[0] == "train":
        train_args = parse_train_args(argv[1:])
        run_train_processes(train_args)
    elif argv[0] == "validate_pred":
        train_args = parse_validate_pred_args(argv[1:])
        run_validate_pred_processes(train_args)
    elif argv[0] == "plot":
        plot_args = parse_plot_args(argv[1:])
        plot_main(plot_args)
    else:
        raise Exception("wrong command line option")
