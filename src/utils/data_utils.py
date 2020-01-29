"""
utility functions to extract data from pandas dataframe
"""
import numpy as np
import pandas as pd
from typing import List
import datetime
from pathlib import Path
import h5py
from src.schema import Catalog
from src.hdf5 import HDF5File


def get_metadata_start_end(df: pd.DataFrame, station: str, begin: str, end: str) \
        -> np.array:
    """
    Get metadata for every time stamp between to dates for a given station
    :param df: pandas dataframe
    :param station: code of the station
    :param begin: begin time string that can be converted to datetime
    :param end: end time string that can be converted to datetime
    :return: np.array containing metadata for each time
    """
    metadata = []
    t0 = pd.Timestamp(begin)
    while t0 + pd.DateOffset(hours=6) < pd.Timestamp(end):
        clearsky = df.loc[t0, f"{station}_CLEARSKY_GHI"]
        clearsky1 = df.loc[t0 + pd.DateOffset(hour=1),
                           f"{station}_CLEARSKY_GHI"]
        clearsky3 = df.loc[t0 + pd.DateOffset(hour=3),
                           f"{station}_CLEARSKY_GHI"]
        clearsky6 = df.loc[t0 + pd.DateOffset(hour=6),
                           f"{station}_CLEARSKY_GHI"]
        daytime = df.loc[t0, f"{station}_DAYTIME"]
        day_of_year = t0.dayofyear
        hour = t0.hour
        minute = t0.minute
        metadata.append([clearsky, clearsky1, clearsky3, clearsky6,
                         daytime, day_of_year, hour, minute])
        t0 += pd.DateOffset(minutes=15)
    return np.array(metadata)


def get_labels_start_end(df: pd.DataFrame, station: str, begin: str, end: str) \
        -> np.array:
    """
    return GHI values at times t0, t0 + 1 hour, t0 + 3 hours and t0 + 6 hours
    :param df: pandas dataframe
    :param station: code of the station
    :param begin: begin time string that can be converted to datetime
    :param end: end time string that can be converted to datetime
    :return: np.array containing labels for each time
    """
    labels = []
    t0 = pd.Timestamp(begin)
    while t0 + pd.DateOffset(hours=6) < pd.Timestamp(end):
        t0_label = df.loc[t0, f"{station}_GHI"]
        t1_label = df.loc[t0 + pd.DateOffset(hours=1), f"{station}_GHI"]
        t2_label = df.loc[t0 + pd.DateOffset(hours=3), f"{station}_GHI"]
        t3_label = df.loc[t0 + pd.DateOffset(hours=6), f"{station}_GHI"]
        labels.append([t0_label, t1_label, t2_label, t3_label])
        t0 += pd.DateOffset(minutes=15)
    return np.array(labels)


def get_labels_list_datetime(df: pd.DataFrame, target_datetimes: List[datetime.datetime],
                             target_time_offsets: List[datetime.timedelta],
                             station: str) -> np.array:
    """
    This function take the same input that we will receive at test time.
    (see function prepare_dataloader in evaluator.py)
    :param station:
    :param target_datetimes: all datetimes that we want the labels from
    :param df: dataframe of catalog.pkl
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    """
    labels = []
    for begin in target_datetimes:
        t0 = pd.Timestamp(begin)
        label = []
        for offset in target_time_offsets:
            label.append(df.loc[t0 + offset, Catalog.ghi(station)])
        labels.append(label)
    return np.array(labels)


def get_hdf5_samples_list_datetime(
        directory: Path,
        target_datetimes: List[datetime.datetime],
        target_time_offsets: List[datetime.timedelta],
        station: str
):
    for begin in target_datetimes:
        t0 = pd.Timestamp(begin)
        # TODO: handle if the starting time is between 2am and 7:45 am
        hdf5_path = Path(directory, f"{t0.year}.{t0.month}.{t0.day}.0800.h5")
        with h5py.File(hdf5_path, "r") as f_h5:
            h5 = HDF5File(f_h5)
            for channel in ["ch1", "ch2", "ch3", "ch4", "ch6"]:
                hdf5_offsets = HDF5File.get_offsets(t0, target_time_offsets)
                for offset in hdf5_offsets:
                    ch_data = h5.fetch_sample(channel, offset)
                    print(ch_data.shape)
                    # TODO: crop here
                    pass
