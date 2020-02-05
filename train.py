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


def main(model_path: str, config_path: str, plot_loss: bool) -> None:
    """
    Train a model and save the weights
    :param model_path: path where model weigths will be saved
    :param config_path: path to json config file
    :param plot_loss: plot losses at end of training if True
    """
    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    epochs = config["epochs"]
    train_batch_size = config["train_batch_size"]
    val_batch_size = config["val_batch_size"]
    patch_size = (config["patch_size"], config["patch_size"])
    dataframe_path = config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    data_path = config["data_path"]
    assert os.path.isdir(data_path), f"invalid data path: {data_path}"

    # Quick fix to avoid nan, should be handled better
    # TODO make sure training and validation examples are valid
    hdf5_paths = dataframe[Catalog.hdf5_8bit_path]
    dataframe = dataframe.fillna(0)
    dataframe[Catalog.hdf5_8bit_path] = hdf5_paths

    train_stations = config["train_stations"]
    val_stations = config["val_stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in config["target_time_offsets"]]

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
                                              batch_size=train_batch_size, test_time=False,
                                              patch_size=patch_size, stations=train_stations)

    val_start_time = config["val_start_bound"]
    val_start_datetime = datetime.datetime.fromisoformat(val_start_time)
    val_end_time = config["val_end_bound"]
    val_end_datetime = datetime.datetime.fromisoformat(val_end_time)
    val_datetimes = []
    while val_start_datetime <= val_end_datetime:
        val_datetimes.append(val_start_datetime)
        val_start_datetime += datetime.timedelta(days=1)

    # TODO always keep the same validation set, only check valid time and only during daytime
    val_data = hdf5_dataloader_list_of_days(dataframe, val_datetimes,
                                            target_time_offsets, data_directory=Path(data_path),
                                            batch_size=val_batch_size, test_time=False,
                                            patch_size=patch_size, stations=val_stations)

    # Here, we assume that the model Class is in a module with the same name and under models
    model_name = config["model_name"]
    model_module = importlib.import_module(f".{model_name}", package="models")
    model = getattr(model_module, model_name)()

    # TODO allow metadata to be configurable
    metadata_len = 4 + len(target_time_offsets)

    model.build([(None, patch_size[0], patch_size[1],
                  5), (None, metadata_len)])

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
    parser.add_argument("-p", "--plot", help="plot the training and validation loss",
                        action="store_true")
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        config_path=args.cfg_path,
        plot_loss=args.plot
    )
