"""
utility functions to extract data from pandas dataframe
"""
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
import datetime
import h5py
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
        df_metadata: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        station: str,
        directory: Optional[str] = None,
) -> Tuple[List[np.array], List[int]]:
    """
    :param df_metadata: catalog.pkl
    :param target_datetimes:
    :param station:
    :param directory: If directory is not provided, use the path from catalog dataframe,
    else use the directory provided but with same filename
    :return: Tuple[patches as np,array, list of index of invalid target_datimes (no picture)]
    """
    # TODO : Deal with changes of file if target_time_offsets is in the future ?
    paths = [df_metadata.at[pd.Timestamp(t), Catalog.hdf5_8bit_path] for t in target_datetimes]
    offsets = [df_metadata.at[pd.Timestamp(t), Catalog.hdf5_8bit_offset] for t in target_datetimes]
    patches = []
    # List of invalid indexes in array, (no data, invalid path, etc.)
    invalids_i = []
    for i, begin in enumerate(target_datetimes):
        if directory is None:
            hdf5_path = paths[i]
        else:
            folder, filename = os.path.split(paths[i])
            hdf5_path = os.path.join(directory, filename)
        with h5py.File(hdf5_path, "r") as f_h5:
            h5 = HDF5File(f_h5)

            lats, lons = None, None
            j = 0
            while lats is None or lons is None:
                lats, lons = h5.fetch_lat_long(j)
                j += 1

            station_coords = h5.get_stations_coordinates(lats, lons, {station: Station.LATS_LONS[station]})
            patch = h5.get_image_patches(offsets[i], station_coords)
            if not patch:
                invalids_i.append(i)
            else:
                patches.append(patch[station])
    return patches, invalids_i
