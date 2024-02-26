import asyncio, json
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from api.influx import InfluxClient
import tensorflow as tf
from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM


# main conf path
CONF_PATH = 'conf.json'

async def fetchnlearn(client : InfluxClient, ml_model_conf : dict):

    try:
        while True:

            await asyncio.sleep(float(ml_model_conf["sleep_dur"]))

            # fetch data
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
            
            # make predictions and push them to the database
            y_points_standard = np.linspace(
                start=y_points[0]*key_scale-(key_mean*key_scale),
                stop=y_points[1]*key_scale-(key_mean*key_scale),
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

    except asyncio.CancelledError:
        pass
    finally:
        logger.warning(f"[live learning server] Stopping fetch and learn task")


async def main():

    # load training config
    with open(CONF_PATH) as json_file:
        config = json.load(json_file)
    logger.info(
        f"Params: \n {config}"
    )
    ml_model_conf = config["ml_model"]
    influx_config = config["influxdb"]

    # connect to influxDB
    client = InfluxClient(influx_config["url"], influx_config["token"], influx_config["bucket"], influx_config["org"], influx_config["read_point_name"])

    task = asyncio.create_task(
        fetchnlearn(client, ml_model_conf)
    )
    await asyncio.gather(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt outside of asyncio.run
        pass
