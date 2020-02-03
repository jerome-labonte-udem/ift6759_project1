import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any, AnyStr, Tuple
import datetime
from src.schema import Station
from src.utils.data_utils import (
    get_labels_list_datetime, get_hdf5_samples_from_day, random_timestamps_from_day, get_metadata_list_datetime
)


def hdf5_dataloader_list_of_days(
        dataframe: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        target_time_offsets: List[datetime.timedelta],
        batch_size: int,
        data_directory: str,
        test_time: bool,
        stations: Dict = None,
        config: Dict[AnyStr, Any] = None,
        patch_size: Tuple[int, int] = (32, 32)
) -> tf.data.Dataset:
    """
    * Train time *: Dataloader that takes as argument a list of days
    (datetime.datetime but where the only thing that matters is the path to the hdf5_file).
    We then sample "batch_size" timestamps from that day, and get a total of
    num_stations * batch_size samples for that day
    * Test time *: Take as input a list of target_datetimes (same as required by script in evaluator.py)
    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
        target_datetimes:
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.
        batch_size: Samples per batch. -- The real batch_size will be num_stations * batch_size --
        data_directory: Provide a data_directory if the directory is not the same as the paths from the dataframe
        test_time: if test_time, return None as target
        patch_size:
    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """
    if stations is None:
        stations = Station.COORDS
    else:
        stations = {k: Station.COORDS[k] for k in stations.keys()}

    def data_generator():
        if test_time:  # Directly use the provided list of datetimes
            for batch_idx in range(0, len(target_datetimes)//batch_size + 1):
                batch_of_datetimes = target_datetimes[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                samples, invalids_i = get_hdf5_samples_from_day(dataframe, batch_of_datetimes,
                                                                directory=data_directory, patch_size=patch_size,
                                                                stations=stations)
                # Remove invalid indexes so that len(targets) == len(samples)
                # Delete them in reverse order so that you don't throw off the subsequent indexes.
                # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time/41079803
                for index in sorted(invalids_i, reverse=True):
                    del batch_of_datetimes[index]
                metadata = get_metadata_list_datetime(dataframe, batch_of_datetimes,
                                                      target_time_offsets, stations)
                targets = tf.zeros(shape=(len(batch_of_datetimes) * len(stations), len(target_time_offsets)))

                yield (samples, metadata), targets
        else:
            for i in range(0, len(target_datetimes)):
                # Generate randomly batch_size timestamps from that given day
                batch_of_datetimes = random_timestamps_from_day(dataframe, target_datetimes[i], batch_size)
                samples, invalids_i = get_hdf5_samples_from_day(dataframe, batch_of_datetimes,
                                                                directory=data_directory, patch_size=patch_size,
                                                                stations=stations)
                # Remove invalid indexes so that len(targets) == len(samples)
                # Delete them in reverse order so that you don't throw off the subsequent indexes.
                # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time/41079803
                for index in sorted(invalids_i, reverse=True):
                    del batch_of_datetimes[index]
                metadata = get_metadata_list_datetime(dataframe, batch_of_datetimes,
                                                      target_time_offsets, stations)
                if test_time:
                    targets = tf.zeros(shape=(len(batch_of_datetimes) * len(stations), len(target_time_offsets)))
                else:
                    targets, invalid_idx_t = get_labels_list_datetime(dataframe, batch_of_datetimes,
                                                                      target_time_offsets, stations)
                    # Remove samples at indexes of invalid targets
                    for index in sorted(invalid_idx_t, reverse=True):
                        del samples[index]
                yield (samples, metadata), targets

    metadata_len = 4 + len(target_time_offsets)
    target_len = len(target_time_offsets)
    # TODO: Check if that's how prefetch should be used
    data_loader = tf.data.Dataset.from_generator(
        data_generator, output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(((None, patch_size[0], patch_size[1], 5), (None, metadata_len)), (None, target_len))
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return data_loader