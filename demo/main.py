# example: python main.py --conf conf_gmm_2h.json

import json, os, sys, threading, copy, time, random
import traceback
import multiprocessing
import multiprocessing.context as ctx
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger.remove()
logger.add(sys.stderr, level="INFO")

def fetchnlearn(influx_config : dict, influx_read_conf: dict, ml_model_conf : dict):
    from api.influx import InfluxClient
    import tensorflow as tf
    from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM

    logger.info("Strating fetch and learn thread")
    
    # connect to influxDB
    client = InfluxClient(influx_config["url"], influx_config["token"], influx_read_conf["bucket"], influx_config["org"], influx_read_conf["read_point_name"])

    MODEL_JSON = ml_model_conf['model_json_file']
    MODEL_H5 = ml_model_conf['model_h5_file']
    
    try:
        while True:

            # fetch data
            y_label = influx_read_conf["y_label"]
            if "dataset_dur" in ml_model_conf:
                df_train = client.get_recent_samples_dur(timedelta(minutes=ml_model_conf["dataset_dur"]["minutes"],seconds=ml_model_conf["dataset_dur"]["seconds"]),field=y_label)
            else:
                df_train = client.get_latest_samples_num(ml_model_conf["training_params"]["dataset_size"])
            if len(df_train) < ml_model_conf["training_params"]["dataset_size"]:
                logger.warning(f'Requested number of samples: {ml_model_conf["training_params"]["dataset_size"]}, received: {len(df_train)}')
                continue
            
            logger.info(f"Number of training samples: {len(df_train)}")
            # shuffle the data
            df_train.sample(replace=True, frac=1)

            # get training parameters
            training_params = ml_model_conf["training_params"]
            model_type = ml_model_conf["type"]
            training_rounds = training_params["rounds"]
            batch_size = training_params["batch_size"]
            strdtype = "float64"

            # dataset pre process
            offset = df_train[y_label].mean()
            scale  = df_train[y_label].std(ddof=0)
            df_train["y_input"] = (df_train[y_label] - offset) / scale    # z-scores
            logger.info(f"Offset: {offset}, scale: {scale}")

            # initiate the non conditional predictor
            if model_type == "gmm":
                model = GaussianMM(
                    centers=ml_model_conf["centers"],
                    dtype=strdtype,
                    bayesian=ml_model_conf["bayesian"]
                )
            elif model_type == "gmevm":
                model = GaussianMixtureEVM(
                    centers=ml_model_conf["centers"],
                    dtype=strdtype,
                    bayesian=ml_model_conf["bayesian"]
                )

            X = None
            Y = df_train.y_input

            steps_per_epoch = len(df_train) // batch_size

            for idx, round_params in enumerate(training_rounds):

                logger.info(
                    "Training session "
                    + f"{idx+1}/{len(training_rounds)} with {round_params}, "
                    + f"steps_per_epoch: {steps_per_epoch}, batch size: {batch_size}"
                )

                model.training_model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=round_params["learning_rate"],
                    ),
                    loss=model.loss,
                )

                Xnp = np.zeros(len(Y))
                Ynp = np.array(Y)
                model.training_model.fit(
                    x=[Xnp, Ynp],
                    y=Ynp,
                    steps_per_epoch=steps_per_epoch,
                    epochs=round_params["epochs"],
                    verbose=0,
                )

            # training done, save the model
            model_conf = {"key_mean":offset, "type":model_type, "key_scale":scale}
            model.save(MODEL_H5)
            with open(MODEL_JSON, "w") as write_file:
                json.dump(model_conf, write_file, indent=4)
            
            logger.info(f"model trained and saved to {MODEL_H5} and {MODEL_JSON} with {model_conf}")
            time.sleep(float(ml_model_conf["sleep_dur_learn"]))

    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        logger.warning(f"[live learning server] Stopping fetch and learn task")


def pushtodb(influx_config : dict, influx_write_config : dict, ml_model_conf : dict):
    from api.influx import InfluxClient
    import tensorflow as tf
    from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM

    logger.info("Strating push to db thread")

    # connect to influxDB
    client = InfluxClient(influx_config["url"], influx_config["token"], influx_write_config["bucket"], influx_config["org"], influx_write_config["write_point_name"])

    y_points = influx_write_config["y_points"]
    write_point_name = influx_write_config["write_point_name"]
    quantiles = np.array(influx_write_config["quantiles"])

    MODEL_JSON = ml_model_conf['model_json_file']
    MODEL_H5 = ml_model_conf['model_h5_file']

    try:
        while True:
            # Set seeds at start of each iteration
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)

            # get the trained model if available
            with open(MODEL_JSON, 'r') as f:
                try:
                    info_dict = json.load(f)
                    offset = float(info_dict["key_mean"])
                    scale = float(info_dict["key_scale"])
                    model_type = info_dict["type"]
                    if model_type == "gmm":
                        model = GaussianMM(h5_addr=MODEL_H5)
                    elif model_type == "gmevm":
                        model = GaussianMixtureEVM(h5_addr=MODEL_H5)
                    else:
                        model = None
                except:
                    model = None
            if not model:
                logger.warning("no model available to read")
                time.sleep(float(ml_model_conf["sleep_dur_dbpush"]))
                continue

            # make predictions and push them to the database
            y = np.linspace(
                start=y_points[0],
                stop=y_points[1],
                num=y_points[2]
            )
            y = np.array(y, dtype=np.float64)
            y_transformed = (y - offset) / scale
            # define y numpy list
            y_transformed = np.array(y_transformed, dtype=np.float64)
            #y = y.clip(min=0.00)
            prob, logprob, cdf = model.prob_batch(y_transformed)
            logccdf = np.log10(np.clip(1.0 - cdf, 1e-15, 1.0))
            res_df = pd.DataFrame({
                'y': y, 
                'prob': prob, 
                'logprob': logprob, 
                'cdf': cdf,
                'ccdf' : 1.0-cdf,
                'logccdf' : np.log10(1.0-cdf)
            })
            #print(res_df)
            logger.debug(f"prediction probability result:\n{res_df}")
            client.push_dataframe(res_df, write_point_name + "_probs")

            # find quantiles
            # quantiles = [0.9, 0.99, 0.999, 0.9999]
            logcquat = np.log10(1.0 - quantiles) # will be [-1 , -2, -3, -4]
            logccdf = np.log10(1.0-cdf)

            # Find indices of closest logccdf values to each logcquat
            indices = [np.abs(logccdf - val).argmin() for val in logcquat]
            y_quantiles = [y[i] for i in indices]
            res_df = pd.DataFrame({
                'y': y_quantiles,
                'quantile': quantiles
            })
            logger.debug(f"prediction quantile result:\n{res_df}")
            client.push_dataframe(res_df, write_point_name + "_quants")

            logger.info(f"Pushed predictions to db")

            time.sleep(float(ml_model_conf["sleep_dur_dbpush"]))

    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        logger.warning(f"[live learning server] Stopping push to db thread")

def main():

    # main conf path
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run with custom config path")
    parser.add_argument('--conf', type=str, default='conf.json', help='Path to configuration file')
    args = parser.parse_args()
    config_file_path = args.conf
    
    logger.info(f"Loading config from {config_file_path}")
    with open(config_file_path) as json_file:
        config = json.load(json_file)


    logger.info(
        f"Params: \n {config}"
    )
    ml_model_conf = config["ml_model"]
    influx_config = config["influxdb"]
    influx_read_config = config["influxdb-read"]
    influx_write_config = config["influxdb-write"]
   
    # fix the files for the models
    MODEL_JSON = ml_model_conf['model_json_file']
    MODEL_H5 = ml_model_conf['model_h5_file']
    # delete the old files and create new ones
    if os.path.exists(MODEL_H5):
        os.remove(MODEL_H5)
    if os.path.exists(MODEL_JSON):
        os.remove(MODEL_JSON)
    os.mknod(MODEL_H5)
    os.mknod(MODEL_JSON)

    try:
        learn_process = multiprocessing.Process(target=fetchnlearn, args=(influx_config,influx_read_config,ml_model_conf),daemon=True)
        push_process = multiprocessing.Process(target=pushtodb, args=(influx_config,influx_write_config,ml_model_conf),daemon=True)

        learn_process.start()
        push_process.start()

        learn_process.join()
        push_process.join()

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        learn_process.terminate()
        push_process.terminate()
    else:
        logger.info("Termination")
        learn_process.terminate()
        push_process.terminate()


if __name__ == "__main__":
    main()
