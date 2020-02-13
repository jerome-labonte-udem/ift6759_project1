"""
Script to train a model and save the weights in a specific location
"""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.data_pipeline import hdf5_dataloader_list_of_days
from src.schema import Catalog

# Directory to save logs for Tensorboard
LOG_DIR = os.path.join("logs", "fit")

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)


def get_callbacks_tensorboard(compile_params: Dict, model_name: str, train_batch_size: int, val_batch_size: int,
                              patch_size: Tuple[int, int]) -> List:
    log_file = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_file,
        histogram_freq=1
    )

    hparams = {
        "model_name": model_name,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "patch_size": patch_size[0],  # type "tuple" is unsupported
    }
    hparams.update(compile_params)

    return [
        tensorboard_callback,
        hp.KerasCallback(log_file, hparams),  # log hparams
    ]


def main(model_path: str, config_path: str, valid_config_path: str, plot_loss: bool) -> None:
    """
    Train a model and save the weights
    :param valid_config_path: path to config file that contains target_datetimes
    :param model_path: path where model weigths will be saved
    :param config_path: path to json config file
    :param plot_loss: plot losses at end of training if True
    """
    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    assert os.path.isfile(valid_config_path), f"invalid valid config file: {valid_config_path}"
    with open(valid_config_path, "r") as config_file:
        valid_config = json.load(config_file)

    epochs = config["epochs"]
    train_batch_size = config["train_batch_size"]
    val_batch_size = config["val_batch_size"]
    patch_size = (config["patch_size"], config["patch_size"])

    dataframe_path = config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"

    data_path = config["data_path"]
    assert os.path.isdir(data_path), f"invalid data path: {data_path}"

    train_stations = config["train_stations"]
    val_stations = config["val_stations"]

    dataframe = pd.read_pickle(dataframe_path)
    # add invalid attribute to datetime if t0 is invalid
    dataframe = Catalog.add_invalid_t0_column(dataframe)
    # Put all GHI values during nightime to 0.0
    for station in train_stations.keys():
        dataframe.loc[dataframe[f"{station}_DAYTIME"] == 0, [f"{station}_GHI"]] = 0

    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in config["target_time_offsets"]]
    previous_time_offsets = [-pd.Timedelta(d).to_pytimedelta() for d in config["previous_time_offsets"]]

    start_time = config["start_bound"]
    start_datetime = datetime.datetime.fromisoformat(start_time)
    end_time = config["end_bound"]
    end_datetime = datetime.datetime.fromisoformat(end_time)
    train_datetimes = []
    while start_datetime <= end_datetime:
        train_datetimes.append(start_datetime)
        start_datetime += datetime.timedelta(days=1)
    train_data = hdf5_dataloader_list_of_days(dataframe, train_datetimes,
                                              target_time_offsets, data_directory=Path(data_path),
                                              batch_size=train_batch_size, subset="train",
                                              patch_size=patch_size, stations=train_stations,
                                              previous_time_offsets=previous_time_offsets)

    val_data = hdf5_dataloader_list_of_days(
        dataframe, valid_config["target_datetimes"], target_time_offsets, data_directory=Path(data_path),
        batch_size=val_batch_size, subset="valid", patch_size=patch_size, stations=val_stations,
        previous_time_offsets=previous_time_offsets
    )

    # Here, we assume that the model Class is in a module with the same name and under models
    model_name = config["model_name"]
    model_module = importlib.import_module(f".{model_name}", package="models")
    timesteps = len(previous_time_offsets)
    target_len = len(target_time_offsets)
    inp_img_seq = tf.keras.layers.Input((timesteps, 32, 32, 5))
    inp_metadata_seq = tf.keras.layers.Input((timesteps, 5))
    inp_future_metadata = tf.keras.layers.Input(target_len)
    inp_shapes = [inp_img_seq, inp_metadata_seq, inp_future_metadata]
    model = getattr(model_module, model_name)()
    model(inp_shapes)
    print(model.summary())

    # model.build([(None, patch_size[0], patch_size[1],
    #              5), (None, metadata_len)])

    compile_params = config["compile_params"]
    model.compile(**compile_params)

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=get_callbacks_tensorboard(
            compile_params, model_name, train_batch_size, val_batch_size, patch_size
        )
    )

    model.save_weights(model_path)

    # TODO save results in some way, probably will be done with Tensorboard

    if plot_loss:
        plt.plot(range(1, epochs + 1), history.history['loss'], label='train_loss')
        plt.plot(range(1, epochs + 1), history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="path where the model should be saved")
    parser.add_argument("--cfg_path", type=str,
                        help="path to the JSON config file used to define train parameters")
    parser.add_argument("--valid_config_path", type=str,
                        help="path to the JSON config file of the validation target_datetimes")
    parser.add_argument("-p", "--plot", help="plot the training and validation loss",
                        action="store_true")
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        config_path=args.cfg_path,
        valid_config_path=args.valid_config_path,
        plot_loss=args.plot
    )
