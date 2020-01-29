from src.utils.data_utils import get_labels_list_datetime, get_hdf5_samples_list_datetime
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Tuple, Any, AnyStr
import datetime
import random


def hdf5_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        stations: Dict[AnyStr, Tuple[float, float, float]],
        target_time_offsets: List[datetime.timedelta],
        batch_size: int,
        data_directory: str,
        test_time: bool,
        config: Dict[AnyStr, Any] = None
) -> tf.data.Dataset:
    """
    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
        target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
            The ordering of this list is important, as each element corresponds to a sequence of GHI values
            to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
            which are added to each timestamp (T=0) in this datetimes list.
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.
        batch_size:
        data_directory:
        test_time: if test_time, return None as target
    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """
    def data_generator():
        for i in range(0, len(target_datetimes), batch_size):
            # TODO: How do we choose the station ? For now random sampling
            station_str = random.choice(list(stations.keys()))
            print(f"Sampled station {station_str}")

            batch_of_datetimes = target_datetimes[i:i + batch_size]
            samples, invalids_i = get_hdf5_samples_list_datetime(dataframe, batch_of_datetimes, station_str,
                                                                 directory=data_directory)
            # TODO: use invalids_i to match samples with targets
            if test_time:
                targets = tf.zeros(shape=batch_size)
            else:
                # TODO: get invalid targets here and remove them
                targets = get_labels_list_datetime(dataframe, batch_of_datetimes,
                                                   target_time_offsets, station_str)
            yield samples, targets

    data_loader = tf.data.Dataset.from_generator(
        data_generator, (tf.float32, tf.float32)
    )
    return data_loader
