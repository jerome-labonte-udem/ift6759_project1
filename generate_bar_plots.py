import argparse
import os

import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing

from src.evaluator import generate_all_predictions, parse_gt_ghi_values, parse_nighttime_flags
from src.schema import Station, get_target_time_offsets
# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)


def parse_cloudiness_flags(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        cloudiness: str
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_CLOUDINESS"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(station_flags.iloc[station_flags.index.get_loc(index)] == cloudiness)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def parse_months(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        month: int
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in dataframe.index:
                    seq_vals.append(index.month == month)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def main(cfg_path: str):
    assert os.path.isfile(cfg_path), f"invalid config file path: {cfg_path}"
    with open(cfg_path, "r") as config_file:
        config = json.load(config_file)
    assert "model_name" in config.keys(), "model_name not found in config file"
    assert "target_datetimes" in config.keys(), "target_datetimes not found in config file"
    assert "previous_time_offsets" in config.keys(), "previous_time_offsets not found in config file"
    assert "saved_weights_path" in config.keys(), "saved_weights_path not found in config file"
    assert "data_directory" in config.keys(), "data_directory not found in config file"
    assert os.path.isdir(config["data_directory"]), f"invalid data_directory: {cfg_path['data_directory']}"
    assert "dataframe_path" in config.keys(), "dataframe_path not found in config file"
    assert os.path.isfile(config["dataframe_path"]), f"invalid dataframe path: {config['dataframe_path']}"
    assert "output_directory" in config.keys(), "output_directory not found in config file"
    output_dir = config["output_directory"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions and graphs at {output_dir}")

    dataframe_path = config["dataframe_path"]
    dataframe = pd.read_pickle(dataframe_path)

    target_datetimes = [datetime.datetime.fromisoformat(d) for d in config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = Station.COORDS
    target_time_offsets = get_target_time_offsets()
    predictions = generate_all_predictions(target_stations, target_datetimes,
                                           target_time_offsets, dataframe, config, config["data_directory"])
    model_name = config["model_name"]
    with open(os.path.join(output_dir, f"{model_name}_preds.txt"), "w") as fd:
        for pred in predictions:
            fd.write(",".join([f"{v:0.03f}" for v in pred.tolist()]) + "\n")
    predictions = predictions.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    gt = parse_gt_ghi_values(target_stations, target_datetimes, target_time_offsets, dataframe)
    gt = gt.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))

    day = parse_nighttime_flags(target_stations, target_datetimes, target_time_offsets, dataframe)
    day = day.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))

    squared_errors = np.square(predictions - gt)

    # Plot RMSE by cloudiness
    cloudiness_rmse = []
    cloudiness_std = []
    cloudiness_values = ['clear', 'cloudy', 'slightly cloudy', 'variable']
    for cloudiness in cloudiness_values:
        mask = parse_cloudiness_flags(target_stations, target_datetimes, target_time_offsets, dataframe, cloudiness)
        mask = mask.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
        rmse = np.sqrt(squared_errors[~np.isnan(gt) & mask & day]).mean()
        std = np.sqrt(squared_errors[~np.isnan(gt) & mask & day]).std()
        cloudiness_rmse.append(rmse)
        cloudiness_std.append(std)
    N = 4
    ind = np.arange(N)
    width = 0.35

    fig = plt.figure()
    plt.bar(ind, cloudiness_rmse, width, yerr=cloudiness_std)

    plt.ylabel('RMSE')
    plt.title(f'{model_name} - RMSE by cloudiness factor')
    plt.xticks(ind, cloudiness_values)
    plt.yticks(np.arange(0, 200, 20))
    plt.savefig(os.path.join(output_dir, f'{model_name}_RMSE_by_cloudiness.png'))
    plt.close(fig)

    # Plot RMSE by month
    fig = plt.figure()
    month_rmse = []
    month_std = []
    month_values = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for month in range(1, 13):
        mask = parse_months(target_stations, target_datetimes, target_time_offsets, dataframe, month)
        mask = mask.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
        rmse = np.sqrt(squared_errors[~np.isnan(gt) & mask & day]).mean()
        std = np.sqrt(squared_errors[~np.isnan(gt) & mask & day]).std()
        month_rmse.append(rmse)
        month_std.append(std)

    N = 12
    ind = np.arange(N)
    width = 0.35

    plt.bar(ind, month_rmse, width, yerr=month_std)

    plt.ylabel('RMSE')
    plt.title(f'{model_name} - RMSE by month')
    plt.xticks(ind, month_values)
    plt.yticks(np.arange(0, 200, 20))
    plt.savefig(os.path.join(output_dir, f'{model_name}_RMSE_by_month.png'))
    plt.close(fig)

    # Plot RMSE by station
    fig = plt.figure()
    station_rmse = []
    station_std = []
    station_names = target_stations.keys()
    for i, station_name in enumerate(station_names):
        rmse = np.sqrt(squared_errors[i][~np.isnan(gt[i]) & day[i]]).mean()
        std = np.sqrt(squared_errors[i][~np.isnan(gt[i]) & day[i]]).std()
        station_rmse.append(rmse)
        station_std.append(std)
        print(station_name, rmse, std)

    N = len(station_names)
    ind = np.arange(N)
    width = 0.35

    plt.bar(ind, station_rmse, width, yerr=station_std)

    plt.ylabel('RMSE')
    plt.title(f'{model_name} - RMSE by station')
    plt.xticks(ind, station_names)
    plt.yticks(np.arange(0, 200, 20))
    plt.savefig(os.path.join(output_dir, f'{model_name}_RMSE_by_station.png'))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    args = parser.parse_args()
    main(
        cfg_path=args.cfg_path
    )
