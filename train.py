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

from src.extract_tf_record import tfrecord_dataloader
from src.schema import Catalog

# Directory to save logs for Tensorboard
LOG_DIR = os.path.join("logs", "fit")

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


def get_callbacks_tensorboard(compile_params: Dict, model_name: str, train_batch_size: int, val_batch_size: int,
                              save_dir: str, patch_size: Tuple[int, int]) -> List:
    log_file = os.path.join(save_dir, LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Saving logs to path {log_file}")
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


def main(save_dir: str, config_path: str, data_path: str, plot_loss: bool) -> None:
    """
    Train a model and save the weights
    :param data_path: directory of tfrecords which includes train and validation folders
    :param save_dir: path to directory to save weights and logs
    :param config_path: path to json config file
    :param plot_loss: plot losses at end of training if True
    """
    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    print(f"Data path is at {data_path}")
    assert os.path.isdir(data_path), f"invalid data_path directory: {data_path}"

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving model and logfiles at {save_dir}")

    epochs = config["epochs"]
    train_batch_size = config["train_batch_size"]
    val_batch_size = config["val_batch_size"]
    patch_size = (config["patch_size"], config["patch_size"])

    dataframe_path = config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"

    train_stations = config["train_stations"]

    dataframe = pd.read_pickle(dataframe_path)
    # add invalid attribute to datetime if t0 is invalid
    dataframe = Catalog.add_invalid_t0_column(dataframe)
    # Put all GHI values during nightime to 0.0
    for station in train_stations.keys():
        dataframe.loc[dataframe[f"{station}_DAYTIME"] == 0, [f"{station}_GHI"]] = 0

    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in config["target_time_offsets"]]
    previous_time_offsets = [-pd.Timedelta(d).to_pytimedelta() for d in config["previous_time_offsets"]]

    model_name = config["model_name"]
    if "seq_len" in config.keys():
        seq_len = config["seq_len"]
    else:
        seq_len = None
    is_cnn = model_name == "CNN2D" or model_name == "VGG2D"
    rotate_imgs = bool(config["rotate_imgs"]) if "rotate_imgs" in config else False
    print(f"Data augmentation: Rotating images = {rotate_imgs}")

    train_data = tfrecord_dataloader(Path(data_path, "train"), is_cnn, patch_size[0], rotate_imgs, seq_len)
    val_data = tfrecord_dataloader(Path(data_path, "validation"), is_cnn, patch_size[0], rotate_imgs, seq_len)

    # Here, we assume that the model Class is in a module with the same name and under models
    model_module = importlib.import_module(f".{model_name}", package="models")
    timesteps = len(previous_time_offsets)
    target_len = len(target_time_offsets)
    inp_img_seq = tf.keras.layers.Input((timesteps, patch_size[0], patch_size[1], 5))
    inp_metadata_seq = tf.keras.layers.Input((timesteps, 5))
    inp_future_metadata = tf.keras.layers.Input(target_len)
    inp_shapes = [inp_img_seq, inp_metadata_seq, inp_future_metadata]
    model = getattr(model_module, model_name)()
    model(inp_shapes)
    print(model.summary())

    compile_params = config["compile_params"]
    model.compile(**compile_params)

    # Saves only best model for now, could be used to saved every n epochs
    model_dir = os.path.join(save_dir, config["saved_weights_path"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"weights.{model_name}")
    print(f"Saving model {model_name} to path = {model_path}")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min',
                                                          verbose=1, save_best_only=True, save_weights_only=True)

    # Stops training when validation accuracy does not go down for "patience" epochs.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                      patience=10, verbose=1)
    tb_callbacks = get_callbacks_tensorboard(
        compile_params, model_name, train_batch_size, val_batch_size, save_dir, patch_size
    )

    history = model.fit(
        train_data.batch(batch_size=train_batch_size),
        epochs=epochs,
        verbose=1,
        validation_data=val_data.batch(batch_size=val_batch_size),
        callbacks=[*tb_callbacks, model_checkpoint, early_stopping]
    )

    if plot_loss:
        completed_epochs = len(history.history['val_loss'])
        plt.plot(range(1, completed_epochs + 1), history.history['loss'][:completed_epochs], label='train_loss')
        plt.plot(range(1, completed_epochs + 1), history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                        help="path of directory where model and logfiles should be saved")
    parser.add_argument("--cfg_path", type=str,
                        help="path to the JSON config file used to define train parameters")
    parser.add_argument("--data_path", type=str, help="directory of the data")
    parser.add_argument("-p", "--plot", help="plot the training and validation loss",
                        action="store_true")
    args = parser.parse_args()
    main(
        save_dir=args.save_dir,
        config_path=args.cfg_path,
        data_path=args.data_path,
        plot_loss=args.plot
    )
