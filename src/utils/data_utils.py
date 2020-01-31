"""
utility functions to extract data from pandas dataframe
"""
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
from _collections import OrderedDict
import datetime
import h5py
import random
from src.schema import Catalog, Station
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


def get_labels_list_datetime(
        df: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        target_time_offsets: List[datetime.timedelta],
        stations: OrderedDict
) -> np.array:
    """
    This function take the same input that we will receive at test time.
    (see function prepare_dataloader in evaluator.py)
    :param stations:
    :param target_datetimes: all datetimes that we want the labels from
    :param df: dataframe of catalog.pkl
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    :return (len(target_datetimes) * 7) labels where each label is 4 GHI values
    """
    labels = []
    for begin in target_datetimes:
        t0 = pd.Timestamp(begin)
        label = []
        for station in stations.keys():
            for offset in target_time_offsets:
                label.append(df.loc[t0 + offset, Catalog.ghi(station)])
            labels.append(label)
    return np.array(labels)


def get_hdf5_samples_from_day(
        df: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        directory: Optional[str] = None,
) -> Tuple[List[np.array], List[int]]:
    """
    :param target_datetimes:
    :param df: catalog.pkl
    :param directory: If directory is not provided (None), use the path from catalog dataframe,
    else use the directory provided but with same filename
    :return: Tuple[patches as np,array, list of index of invalid target_datimes (no picture)]
    """
    # path should be the same for all datetimes
    path = df.at[pd.Timestamp(target_datetimes[0]), Catalog.hdf5_8bit_path]
    sample_indexes = [df.at[pd.Timestamp(t), Catalog.hdf5_8bit_offset] for t in target_datetimes]
    patches = []
    # List of invalid indexes in array, (no data, invalid path, etc.)
    invalids_i = []
    if directory is None:
        hdf5_path = path
    else:
        folder, filename = os.path.split(path)
        hdf5_path = os.path.join(directory, filename)
    with h5py.File(hdf5_path, "r") as f_h5:
        h5 = HDF5File(f_h5)
        for i, index in enumerate(sample_indexes):
            patches_index = h5.get_image_patches(index, Station.COORDS)
            if not patches_index:
                invalids_i.append(i)
            else:
                patches.extend(patches_index)
    return patches, invalids_i


def random_timestamps_from_day(df: pd.DataFrame, target_day: datetime.datetime,
                               batch_size) -> List[datetime.datetime]:
    """
    Randomly sample batch_size timestamps from the target_day
    :param df: catalog.pkl
    :param target_day: datetime from the day of the hdf5 file we want to open (hour doesn't matter)
    :param batch_size: number of timestamps we want to sample
    :return: list of random timestamps from given day
    """
    path_day = df.at[target_day, Catalog.hdf5_8bit_path]
    all_photos_day = df.loc[df[Catalog.hdf5_8bit_path] == path_day]
    # TODO: Filter here some timestamps that we don't want (e.g. during the night) ??
    random_timestamps = random.choices(all_photos_day.index, k=batch_size)
    return random_timestamps


def filter_catalog(df: pd.DataFrame, remove_invalid_labels: bool) -> pd.DataFrame:
    """
    Remove Invalid hours where photos are never available for the satellite
    Also remove rows where the path to hdf5 is invalid
    :param remove_invalid_labels: If true, we also remove entries that have nan at some GHI column
    :param df: catalog.pkl
    :return: catalog.pkl without invalid entries
    """
    # TODO: Maybe there is a way to speed this up ? But only takes 1-2 seconds
    invalid_hours = [
        (0, 0), (0, 30), (3, 0), (6, 0), (9, 0),
        (12, 0), (15, 0), (15, 30), (18, 0), (21, 0)
    ]
    for hour, minute in invalid_hours:
        df = df.loc[(df.index.hour != hour) | (df.index.minute != minute)]

    # Remove entries that have nan in the path
    df = df[pd.notnull(df[Catalog.hdf5_8bit_path])]

    if remove_invalid_labels:
        for station in Station.list():
            df = df[pd.notnull(df[Catalog.ghi(station)])]
    return df
