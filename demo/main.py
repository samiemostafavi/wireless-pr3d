import json, os, sys, threading, copy, time
import traceback
import multiprocessing
import multiprocessing.context as ctx
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger.remove()
logger.add(sys.stderr, level="INFO")

# main conf path
CONF_PATH = 'conf.json'
MODEL_JSON = '.model.json'
MODEL_H5 = '.model.h5'

# delete the old files and create new ones
if os.path.exists(MODEL_H5):
    os.remove(MODEL_H5)
if os.path.exists(MODEL_JSON):
    os.remove(MODEL_JSON)
os.mknod(MODEL_H5)
os.mknod(MODEL_JSON)

def fetchnlearn(influx_config : dict, ml_model_conf : dict):
    from api.influx import InfluxClient
    import tensorflow as tf
    from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM

    logger.info("Strating fetch and learn thread")
    
    # connect to influxDB
    client = InfluxClient(influx_config["url"], influx_config["token"], influx_config["bucket"], influx_config["org"], influx_config["read_point_name"])
    
    try:
        while True:

            # fetch data
            if "dataset_dur" in ml_model_conf:
                df_train = client.get_recent_samples_dur(timedelta(minutes=ml_model_conf["dataset_dur"]["minutes"],seconds=ml_model_conf["dataset_dur"]["seconds"]))
            else:
                df_train = client.get_latest_samples_num(ml_model_conf["training_params"]["dataset_size"])
            if len(df_train) < ml_model_conf["training_params"]["dataset_size"]:
                logger.warning(f'Requested number of samples: {ml_model_conf["training_params"]["dataset_size"]}, received: {len(df_train)}')
                continue

            # shuffle the data
            df_train.sample(replace=True, frac=1)

            # get training parameters
            training_params = ml_model_conf["training_params"]
            y_label = ml_model_conf["y_label"]
            model_type = ml_model_conf["type"]
            training_rounds = training_params["rounds"]
            batch_size = training_params["batch_size"]
            key_scale = np.float64(ml_model_conf["scale"])
            y_points = ml_model_conf["y_points"]
            write_point_name = ml_model_conf["write_point_name"]
            strdtype = "float64"

            # dataset pre process
            df_train = df_train[[y_label]]
            df_train["y_input"] = df_train[y_label].apply(lambda x:x*key_scale)
            key_mean = df_train[y_label].mean()
            logger.debug(f"Key mean: {key_mean}")

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
            model_conf = {"key_mean":key_mean, "type":model_type, "key_scale":key_scale}
            model.save(MODEL_H5)
            with open(MODEL_JSON, "w") as write_file:
                json.dump(model_conf, write_file, indent=4)
            
            logger.info("model trained and saved")
            time.sleep(float(ml_model_conf["sleep_dur_learn"]))

    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        logger.warning(f"[live learning server] Stopping fetch and learn task")


def pushtodb(influx_config : dict, ml_model_conf : dict):
    from api.influx import InfluxClient
    import tensorflow as tf
    from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM

    logger.info("Strating push to db thread")

    # connect to influxDB
    client = InfluxClient(influx_config["url"], influx_config["token"], influx_config["bucket"], influx_config["org"], influx_config["read_point_name"])

    y_points = ml_model_conf["y_points"]
    write_point_name = ml_model_conf["write_point_name"]

    try:
        while True:
            # get the trained model if available
            with open(MODEL_JSON, 'r') as f:
                try:
                    info_dict = json.load(f)
                    key_mean = float(info_dict["key_mean"])
                    key_scale = float(info_dict["key_scale"])
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
                logger.warning("no model available to push")
                time.sleep(float(ml_model_conf["sleep_dur_dbpush"]))
                continue

            # make predictions and push them to the database
            y_points_standard = np.linspace(
                start=y_points[0], #*key_scale-(key_mean*key_scale)
                stop=y_points[1], #*key_scale-(key_mean*key_scale)
                num=y_points[2],
            )
            # define y numpy list
            y = np.array(y_points_standard, dtype=np.float64)
            #y = y.clip(min=0.00)
            prob, logprob, cdf = model.prob_batch(y)
            res_df = pd.DataFrame({ 
                'y': y+(key_mean*key_scale), 
                'prob': prob, 
                'logprob': logprob, 
                'cdf': cdf,
                'ccdf' : 1.0-cdf,
                'logccdf' : np.log10(1.0-cdf)
            })
            logger.debug(f"prediction result:\n{res_df}")
            client.push_dataframe(res_df, write_point_name)
            logger.info(f"Pushed a model to db")

            time.sleep(float(ml_model_conf["sleep_dur_dbpush"]))

    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        logger.warning(f"[live learning server] Stopping fetch and learn task")

def main():

    # get standalone env variable
    config_file_path = os.environ.get("CONFIG_FILE_PATH")
    if config_file_path is not None:
        logger.info(f"Loading config from {config_file_path}")
        with open(config_file_path) as json_file:
            config = json.load(json_file)
    else:
        # load config from default path
        logger.info("Loading config from default path")
        with open(CONF_PATH) as json_file:
            config = json.load(json_file)

    logger.info(
        f"Params: \n {config}"
    )
    ml_model_conf = config["ml_model"]
    influx_config = config["influxdb"]

    try:
        learn_process = multiprocessing.Process(target=fetchnlearn, args=(influx_config,ml_model_conf),daemon=True)
        push_process = multiprocessing.Process(target=pushtodb, args=(influx_config,ml_model_conf),daemon=True)

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
