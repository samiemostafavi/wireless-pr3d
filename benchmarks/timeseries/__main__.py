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

from .plot_parquet import parse_plot_parquet_args, run_plot_parquet_processes

# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":

    argv = sys.argv[1:]
    if argv[0] == "plot_parquet":
        train_args = parse_plot_parquet_args(argv[1:])
        run_plot_parquet_processes(train_args)
    else:
        raise Exception("wrong command line option")
