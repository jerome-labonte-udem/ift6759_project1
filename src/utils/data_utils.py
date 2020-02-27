"""
utility functions to extract data from pandas dataframe
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Set, List, Tuple, Optional
from _collections import OrderedDict
import datetime
import h5py
import logging
import json
from src.schema import Catalog, Station
from src.hdf5 import HDF5File


def get_labels_list_datetime(
        df: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        target_time_offsets: List[datetime.timedelta],
        stations: OrderedDict
) -> Tuple[List[List], Set[int]]:
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
    invalid_indexes = set()
    default_ghi = 0
    for i, begin in enumerate(target_datetimes):
        t0 = pd.Timestamp(begin)
        # If one GHI (out of 4) of the time stamp + offsets is invalid -> we remove that sample
        for j, station in enumerate(stations.keys()):
            list_ghi = []
            for offset in target_time_offsets:
                try:
                    ghi = df.loc[t0 + offset, Catalog.ghi(station)]
                    if np.isnan(ghi):
                        invalid_indexes.add(i * len(stations) + j)
                        list_ghi.append(default_ghi)
                    else:
                        list_ghi.append(ghi)
                except KeyError as err:
                    # Probably trying to look in 2016 but we don't have access to these GHI values
                    logging.debug(f"KeyError: {err}")
                    invalid_indexes.add(i * len(stations) + j)
                    list_ghi.append(default_ghi)
            labels.append(list_ghi)
    return labels, invalid_indexes


def get_metadata(df: pd.DataFrame, target_datetimes: List[datetime.datetime],
                 past_time_offsets: List[datetime.timedelta],
                 target_time_offsets: List[datetime.timedelta], stations: OrderedDict) -> Tuple[List[List], List]:
    """
    Get past and future metadata for every datetime in given list
    :param df: pandas dataframe
    :param target_datetimes: all datetimes that we want metadata for
    :param past_time_offsets: the list of timedeltas that we want to get data back from T0
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    :param stations: dict station
    :return: list of np.array containing past metadata and future metadata for each datetime
    """
    # This constant was computed once on the 2010-2015 data
    # MAX_CLEARSKY_GHI = 1045.112902
    past_metadatas = []
    future_metadatas = []
    place_holder_past = [0.] * 5
    place_holder_future = 0
    for begin in target_datetimes:
        t0 = pd.Timestamp(begin)
        for station in stations.keys():
            metadata_sequence = []
            for offset in past_time_offsets:
                try:
                    metadata = [
                        df.loc[t0 + offset, f"{station}_CLEARSKY_GHI"],
                        df.loc[t0 + offset, f"{station}_DAYTIME"],
                        (t0 + offset).dayofyear / 365 * 2 - 1,
                        (t0 + offset).hour / 24 * 2 - 1,
                        (t0 + offset).minute / 60 * 2 - 1
                    ]
                    # rescale to [-1, 1]
                    metadata_sequence.append(metadata)
                except KeyError as err:
                    # If CLEARSKY_GHI not available in df -> GHI not available as well
                    # so these timestamps will be removed from get_labels()
                    # Probably trying to look in 2016 but we don't have access to these values
                    logging.debug(f"KeyError: {err}")
                    metadata_sequence.append(place_holder_past)
            past_metadatas.append(metadata_sequence)

            future_metadata = []
            for offset in target_time_offsets:
                try:
                    future_metadata.append(df.loc[t0 + offset, f"{station}_CLEARSKY_GHI"])
                except KeyError as err:
                    # If CLEARSKY_GHI not available in df -> GHI not available as well
                    # so these timestamps will be removed from get_labels()
                    # Probably trying to look in 2016 but we don't have access to these values
                    logging.debug(f"KeyError: {err}")
                    future_metadata.append(place_holder_future)
            future_metadatas.append(future_metadata)
    return past_metadatas, future_metadatas


def get_hdf5_samples_list_datetime(
        df: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        previous_time_offsets: List[datetime.timedelta],
        test_time: bool,
        patch_size: Tuple[int, int],
        directory: Optional[str] = None,
        stations: OrderedDict = Station.COORDS,
) -> Tuple[List[np.array], List[int], List, List]:
    """
    Open one .hdf5 file for each target_datetime provided
    :param test_time: if test_time, we are never skipping a data point
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
    # min and max of arrays that was used to compress them
    a_min, a_max = [], []
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
            patches_index, min_index, max_index = _get_one_sample(
                h5, target_datetimes[i], sample_offsets[i], test_time, stations,
                patch_size, previous_time_offsets, h5_previous
            )
            if patches_index is None or len(patches_index) == 0:
                invalid_indexes.append(i)
            else:
                patches.extend(patches_index)
                a_min.extend(min_index)
                a_max.extend(max_index)

        if f_h5_before is not None:
            f_h5_before.close()

    return patches, invalid_indexes, a_min, a_max


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
        test_time: bool,
        stations: OrderedDict,
        patch_size: Tuple[int, int],
        previous_time_offsets: List[datetime.timedelta],
        h5_previous: HDF5File = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    patches_index = []
    # Store min and max to recompress array in tfrecords
    min_index, max_index = [], []
    patch = h5.get_image_patches(t0_offset, test_time, stations, patch_size=patch_size)

    if patch is None or len(patch) == 0:  # t0 image is invalid
        return None, None, None

    previous_offsets = HDF5File.get_offsets(
        pd.Timestamp(target_datetime), previous_time_offsets
    )

    for prev_offset, is_previous in previous_offsets:
        if is_previous:
            # Looking in file from day before
            if h5_previous is None:
                patch_prev = None
                # if no image, we use global minimum and global maximum
                min_prev = len(patch) * [list(np.reshape(HDF5File.MIN_CHANNELS, 5))]
                max_prev = len(patch) * [list(np.reshape(HDF5File.MAX_CHANNELS, 5))]
            else:
                patch_prev = h5_previous.get_image_patches(prev_offset, test_time, stations, patch_size=patch_size)
                min_prev = len(patch) * [[h5_previous.orig_min(channel) for channel in HDF5File.CHANNELS]]
                max_prev = len(patch) * [[h5_previous.orig_max(channel) for channel in HDF5File.CHANNELS]]
        else:
            patch_prev = h5.get_image_patches(prev_offset, test_time, stations, patch_size=patch_size)
            min_prev = len(patch) * [[h5.orig_min(channel) for channel in HDF5File.CHANNELS]]
            max_prev = len(patch) * [[h5.orig_max(channel) for channel in HDF5File.CHANNELS]]

        if patch_prev is None or len(patch_prev) == 0:
            # No image available at t0 - offset
            patches_index.insert(len(patches_index) - 1, np.zeros_like(patch))
        else:
            patches_index.insert(len(patches_index) - 1, patch_prev)

        min_index.insert(len(patches_index) - 1, min_prev)
        max_index.insert(len(patches_index) - 1, max_prev)

    patches_index = np.stack(patches_index, axis=-1)

    min_index = np.stack(min_index, axis=-1)
    max_index = np.stack(max_index, axis=-1)
    # output should be [len(stations), len(previous_time_offsets), 1, 1, n_channels]
    for i in [2, 3]:
        min_index = np.expand_dims(min_index, i)
        max_index = np.expand_dims(max_index, i)

    # We want size (len_stations, len(previous_time_offsets), patch_size, patch_size, n_channels)
    patches_index = np.transpose(patches_index, (0, 4, 1, 2, 3))
    return patches_index, min_index, max_index


def generate_random_timestamps_for_validation(
        df: pd.DataFrame,
        n_per_day: int,
        sampling: bool
):
    """
    One time function to generate a validation set of size X
    :param sampling:
    :param df: catalog.pkl
    :param n_per_day: number of timestamps to sample per day of 2015
    :return:
    """
    df = df.loc[df.index.year == 2015]
    df = df.loc[df[Catalog.is_invalid] == 0]
    df = df.groupby([Catalog.hdf5_8bit_path])
    dct = {"target_datetimes": []}
    for path, df_day in df:
        indexes = df_day.index.values
        if sampling:
            # Don't append same datetime X times if less indexes than n_per_day
            n_sample = min(len(indexes), n_per_day)
            samples = [np.datetime_as_string(sample, unit='s') for sample in np.random.choice(indexes, n_sample)]
        else:
            samples = [np.datetime_as_string(sample, unit='s') for sample in indexes]
        # make sure no duplicates cuz days can be pointing to yesterday and today
        for sample in samples:
            if sample not in dct["target_datetimes"]:
                dct["target_datetimes"].append(sample)
    with open(f'valid_datetimes_{len(dct["target_datetimes"])}.json', 'w') as outfile:
        json.dump(dct, outfile, indent=2)


def main():
    data_dir = Path(Path(__file__).parent.parent.parent, "data")
    data_path = Path(data_dir, "catalog.helios.public.20100101-20160101.pkl")
    df = Catalog.add_invalid_t0_column(pd.read_pickle(data_path))
    generate_random_timestamps_for_validation(df, n_per_day=4, sampling=True)


if __name__ == "__main__":
    main()
