"""
Create batches in pickle files
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from src.data_pipeline import hdf5_dataloader_test
from src.schema import Catalog


def main(pickle_path: str,
         data_config_path: str,
         train_datetimes_path: str,
         val_datetimes_path: str) -> None:
    """
    Create one pickle file per minibatch
    :param pickle_path: path to directory where to save the pickles
    :param data_config_path: path to the data_config path
    :param train_datetimes_path: path to file containing the train datetimes
    :param val_datetimes_path: path to file containing the validation_datetimes
    """
    assert os.path.isfile(data_config_path), f"invalid config file: {data_config_path}"
    with open(data_config_path, "r") as config_file:
        config = json.load(config_file)
    assert os.path.isfile(train_datetimes_path), f"invalid train datetimes file: {train_datetimes_path}"
    with open(train_datetimes_path, "r") as train_datetimes_file:
        train_datetimes = json.load(train_datetimes_file)["target_datetimes"]

    assert os.path.isfile(val_datetimes_path), f"invalid val datetimes file: {val_datetimes_path}"
    with open(val_datetimes_path, "r") as val_datetimes_file:
        val_datetimes = json.load(val_datetimes_file)["target_datetimes"]

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

    train_data = hdf5_dataloader_test(dataframe,
                                      train_datetimes,
                                      target_time_offsets=target_time_offsets,
                                      data_directory=Path(data_path),
                                      batch_size=train_batch_size,
                                      subset="valid",
                                      patch_size=patch_size,
                                      stations=train_stations,
                                      previous_time_offsets=previous_time_offsets
                                      )

    val_data = hdf5_dataloader_test(dataframe,
                                    val_datetimes,
                                    target_time_offsets=target_time_offsets,
                                    data_directory=Path(data_path),
                                    batch_size=val_batch_size,
                                    subset="valid",
                                    patch_size=patch_size,
                                    stations=val_stations,
                                    previous_time_offsets=previous_time_offsets
                                    )

    # dump to pickle
    for i, minibatch in enumerate(train_data):
        pickle.dump(minibatch, open(Path(pickle_path, f"train_batch_{i}.pkl"), "wb"))

    for i, minibatch in enumerate(val_data):
        pickle.dump(minibatch, open(Path(pickle_path, f"val_batch_{i}.pkl"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str,
                        help="path where the pickles will be saved")
    parser.add_argument("--cfg_path", type=str,
                        help="path to the JSON data_config file")
    parser.add_argument("--train_datetimes_path", type=str,
                        help="path to the JSON file of train target_datetimes")
    parser.add_argument("--val_datetimes_path", type=str,
                        help="path to the JSON file of the validation target datetimes")
    args = parser.parse_args()
    main(pickle_path=args.pickle_path,
         data_config_path=args.cfg_path,
         train_datetimes_path=args.train_datetimes_path,
         val_datetimes_path=args.val_datetimes_path)
