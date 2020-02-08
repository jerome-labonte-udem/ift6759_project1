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
import logging


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
                    logging.debug(f"KeyError: {err}")
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
    place_holder = 0.0
    for begin in target_datetimes:
        t0 = pd.Timestamp(begin)
        for station in stations.keys():
            metadata = []
            try:
                for offset in target_time_offsets:
                    metadata.append(df.loc[t0 + offset, f"{station}_CLEARSKY_GHI"])
            except KeyError as err:
                # If CLEARSKY_GHI not available in df -> GHI not available as well
                # so these timestamps will be removed from get_labels()
                # Probably trying to look in 2016 but we don't have access to these values
                logging.debug(f"KeyError: {err}")
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
        stations: OrderedDict = Station.COORDS,
        previous_time_offsets: List[datetime.timedelta] = None
) -> Tuple[List[np.array], List[int]]:
    """
    This is for train/validation time only.
    Assume each target datetime has same path in the dataframe (come from same day)
    Get len(target_days) sample from only one .hdf5 file
    :param previous_time_offsets: *Important* Assume Chronological order of deltas (e.g. -12, -6, -3, -1)
    :param stations:
    :param patch_size:
    :param target_datetimes:
    :param df: catalog.pkl
    :param directory: If directory is not provided (None), use the path from catalog dataframe,
    else use the directory provided but with same filename
    :return: Tuple[patches as np,array, list of index of invalid target_datimes (no picture)]
    """
    # path should be the same for all datetimes
    t0 = pd.Timestamp(target_datetimes[0])
    path = df.at[t0, Catalog.hdf5_8bit_path]

    if directory is None:
        hdf5_path = path
    else:
        folder, filename = os.path.split(path)
        hdf5_path = os.path.join(directory, filename)

    sample_offsets = [df.at[pd.Timestamp(t), Catalog.hdf5_8bit_offset] for t in target_datetimes]

    # Make sure the file of previous day exists !
    f_h5_before = _get_path_yesterday(t0, df, path, directory)
    h5_previous = None if f_h5_before is None else HDF5File(f_h5_before)

    patches = []
    # List of invalid indexes in array, (no data, invalid path, etc.)
    invalid_indexes = []

    with h5py.File(hdf5_path, "r") as f_h5:
        h5 = HDF5File(f_h5)
        for i, offset in enumerate(sample_offsets):
            patches_index = _get_one_sample(
                h5, target_datetimes[i], offset, stations, patch_size, previous_time_offsets, h5_previous
            )
            if patches_index is None or len(patches_index) == 0:
                invalid_indexes.append(i)
            else:
                patches.extend(patches_index)

    if f_h5_before is not None:
        f_h5_before.close()

    return patches, invalid_indexes


def get_hdf5_samples_list_datetime(
        df: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        patch_size: Tuple[int, int],
        directory: Optional[str] = None,
        stations: OrderedDict = Station.COORDS,
        previous_time_offsets: List[datetime.timedelta] = None
) -> Tuple[List[np.array], List[int]]:
    """
    Open one .hdf5 file for each target_datetime provided
    :param previous_time_offsets:
    :param stations:
    :param patch_size:
    :param target_datetimes:
    :param df: catalog.pkl
    :param directory: If directory is not provided (None), use the path from catalog dataframe,
    else use the directory provided but with same filename
    :return: Tuple[patches as np,array, list of index of invalid target_datimes (no picture)]
    """
    paths = [df.at[pd.Timestamp(t), Catalog.hdf5_8bit_path] for t in target_datetimes]
    sample_offsets = [df.at[pd.Timestamp(t), Catalog.hdf5_8bit_offset] for t in target_datetimes]
    patches = []
    # List of invalid indexes in array, (no data, invalid path, etc.)
    invalid_indexes = []
    for i, path in enumerate(paths):
        if directory is None:
            hdf5_path = path
        else:
            folder, filename = os.path.split(path)
            hdf5_path = os.path.join(directory, filename)

        # Make sure the file of previous day exists !
        t0 = pd.Timestamp(target_datetimes[i])
        f_h5_before = _get_path_yesterday(t0, df, path, directory)
        h5_previous = None if f_h5_before is None else HDF5File(f_h5_before)

        with h5py.File(hdf5_path, "r") as f_h5:
            h5 = HDF5File(f_h5)
            patches_index = _get_one_sample(
                h5, target_datetimes[i], sample_offsets[i], stations, patch_size, previous_time_offsets, h5_previous
            )
            if patches_index is None or len(patches_index) == 0:
                invalid_indexes.append(i)
            else:
                patches.extend(patches_index)

        if f_h5_before is not None:
            f_h5_before.close()

    return patches, invalid_indexes


def _get_path_yesterday(t0: pd.Timestamp, df: pd.DataFrame, path: str, directory: str) -> Optional[h5py.File]:
    # Make sure the file of previous day exists !
    t_min_24 = t0 - datetime.timedelta(hours=24)
    if t_min_24 in df.index:
        path_day_before = df.at[t_min_24, Catalog.hdf5_8bit_path]
        if directory is None:
            hdf5_path_before = path_day_before
        else:
            folder, filename = os.path.split(path)
            hdf5_path_before = os.path.join(directory, filename)
        if os.path.exists(hdf5_path_before):
            f_h5_before = h5py.File(hdf5_path_before, "r")
        else:
            logging.debug(f"HDF5 file for previous day doesn't exist ! path = {hdf5_path_before}")
            f_h5_before = None
    else:
        logging.debug(f"Day before t0={t0} is not in the dataframe! t-24 = {t_min_24}")
        f_h5_before = None
    return f_h5_before


def _get_one_sample(
        h5: HDF5File,
        target_datetime: datetime.datetime,
        t0_offset: int,
        stations: OrderedDict,
        patch_size: Tuple[int, int],
        previous_time_offsets: List[datetime.timedelta],
        h5_previous: HDF5File = None,
):
    patches_index = []
    patch = h5.get_image_patches(t0_offset, stations, patch_size=patch_size)

    if patch is None or len(patch) == 0:  # t0 image is invalid
        return None
    else:
        if previous_time_offsets is None:
            return patch
        else:
            patches_index.append(patch)

    previous_offsets = HDF5File.get_offsets(
        pd.Timestamp(target_datetime), previous_time_offsets
    )

    for prev_offset, is_previous in previous_offsets:
        if is_previous:
            # Looking in file from day before
            if h5_previous is None:
                patch_prev = None
            else:
                patch_prev = h5_previous.get_image_patches(prev_offset, stations, patch_size=patch_size)
        else:
            patch_prev = h5.get_image_patches(prev_offset, stations, patch_size=patch_size)

        if patch_prev is None or len(patch_prev) == 0:
            # No image available at t0 - offset
            patches_index.insert(len(patches_index) - 1, np.zeros_like(patch))
        else:
            patches_index.insert(len(patches_index) - 1, patch_prev)

    patches_index = np.stack(patches_index, axis=-1)
    # We want size (len_stations, len(previous_time_offsets) + 1, patch_size, patch_size, n_channels)
    patches_index = np.transpose(patches_index, (0, 4, 1, 2, 3))
    return patches_index


def random_timestamps_from_day(df: pd.DataFrame, target_day: datetime.datetime,
                               batch_size) -> List[datetime.datetime]:
    """
    Randomly sample batch_size timestamps from the target_day
    ** filter_timestamps_train_time has to be called on dataframe before entering here **
    :param df: catalog.pkl
    :param target_day: datetime from the day of the hdf5 file we want to open (hour doesn't matter)
    :param batch_size: number of timestamps we want to sample
    :return: list of random timestamps from given day
    """
    path_day = df.at[target_day, Catalog.hdf5_8bit_path]
    photos = df.loc[(df[Catalog.hdf5_8bit_path] == path_day) & (df[Catalog.is_invalid] == 0)]
    random_timestamps = random.choices(photos.index, k=batch_size)
    return random_timestamps
