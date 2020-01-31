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
) -> Tuple[np.array, List[int]]:
    """
    This function take the same input that we will receive at test time.
    (see function prepare_dataloader in evaluator.py)
    :param stations:
    :param target_datetimes: all datetimes that we want the labels from
    :param df: dataframe of catalog.pkl
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    :return Tuple[(len(target_datetimes) * 7) labels where each label is 4 GHI values,
                  list of indexes of invalid time_stamps]
    """
    labels = []
    invalid_indexes = []
    for i, begin in enumerate(target_datetimes):
        t0 = pd.Timestamp(begin)
        # If one GHI (out of 4) of the time stamp + offsets is invalid -> we remove that sample
        for j, station in enumerate(stations.keys()):
            list_ghi = []
            invalid = False
            for offset in target_time_offsets:
                try:
                    ghi = df.loc[t0 + offset, Catalog.ghi(station)]
                    if np.isnan(ghi):
                        invalid_indexes.append(i * len(stations) + j)
                        invalid = True
                        break
                    else:
                        list_ghi.append(ghi)
                except KeyError as err:
                    # Probably trying to look in 2016 but we don't have access to these GHI values
                    print(f"KeyError: {err}")
                    invalid_indexes.append(i * len(stations) + j)
                    invalid = True
                    break
            if not invalid:
                labels.append(list_ghi)
    return np.array(labels), invalid_indexes


def get_metadata_list_datetime(df: pd.DataFrame, target_datetimes: List[datetime.datetime],
                               target_time_offsets: List[datetime.timedelta],
                               stations: OrderedDict) \
        -> List:
    """
    Get metadata for every datetime in given list
    :param df: pandas dataframe
    :param target_datetimes: all datetimes that we want metadata for
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    :param stations: dict station
    :return: np.array containing metadata for each time
    """
    metadatas = []
    place_holder = 0.0  # TODO: fix this
    for begin in target_datetimes:
        t0 = pd.Timestamp(begin)
        for station in stations.keys():
            metadata = []
            try:
                for offset in target_time_offsets:
                    metadata.append(df.loc[t0 + offset, f"{station}_CLEARSKY_GHI"])
            except KeyError as err:
                # TODO: @Marie, add your computation for _CLEARSKY_GHI here.. ?
                # Probably trying to look in 2016 but we don't have access to these values
                print(f"KeyError: {err}")
                metadata.append(place_holder)
            metadata.append(df.loc[t0, f"{station}_DAYTIME"])
            metadata.append(t0.dayofyear)
            metadata.append(t0.hour)
            metadata.append(t0.minute)
            metadatas.append(metadata)
    return metadatas


def get_hdf5_samples_from_day(
        df: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        patch_size: Tuple[int, int],
        directory: Optional[str] = None,
) -> Tuple[List[np.array], List[int]]:
    """
    Get len(target_datetimes) sample from only one .hdf5 file
    :param patch_size:
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
    invalid_indexes = []
    if directory is None:
        hdf5_path = path
    else:
        folder, filename = os.path.split(path)
        hdf5_path = os.path.join(directory, filename)
    with h5py.File(hdf5_path, "r") as f_h5:
        h5 = HDF5File(f_h5)
        for i, index in enumerate(sample_indexes):
            patches_index = h5.get_image_patches(index, Station.COORDS, patch_size=patch_size)
            if not patches_index:
                invalid_indexes.append(i)
            else:
                patches.extend(patches_index)
    return patches, invalid_indexes


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
    photos = df.loc[df[Catalog.hdf5_8bit_path] == path_day]

    # Remove all photos that we know are invalid
    # TODO: Faster/Better way to do this ?
    for hour, minute in Catalog.invalid_hours():
        photos = photos.loc[(photos.index.hour != hour) | (photos.index.minute != minute)]

    # TODO: Filter here some timestamps that we don't want (e.g. during the night) ??
    random_timestamps = random.choices(photos.index, k=batch_size)
    return random_timestamps
