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

from .plot_acf import parse_plot_acf_args, run_plot_acf_processes
from .train_nc_lstm import parse_train_nc_lstm_args, run_train_nc_lstm_processes

# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":

    argv = sys.argv[1:]
    if argv[0] == "plot_acf":
        train_args = parse_plot_acf_args(argv[1:])
        run_plot_acf_processes(train_args)
    elif argv[0] == "train_nc_lstm":
        train_args = parse_train_nc_lstm_args(argv[1:])
        run_train_nc_lstm_processes(train_args)
    else:
        raise Exception("wrong command line option")
