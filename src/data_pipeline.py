import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any, AnyStr, Tuple, Optional
import datetime

from src.hdf5 import HDF5File
from src.schema import Station, get_previous_time_offsets
from src.utils.data_utils import (
    get_labels_list_datetime, get_metadata, get_hdf5_samples_list_datetime
)


def hdf5_dataloader_test(
        dataframe: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        target_time_offsets: List[datetime.timedelta],
        previous_time_offsets: List[datetime.timedelta],
        batch_size: int,
        subset: str,
        stations: Dict = None,
        config: Dict[AnyStr, Any] = None,
        data_directory: Optional[str] = None,
        patch_size: Tuple[int, int] = (32, 32),
        normalize_imgs: bool = True,
) -> tf.data.Dataset:
    """
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
        data_directory: (Optional) Provide a data_directory if the directory is not the same as
        the paths from the dataframe
        subset: "test", "valid", or "train
        patch_size:
        stations:
        normalize_imgs:
        previous_time_offsets: list of timedelta of previous pictures that we want to look at,
        if not provided we only look at t0
    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """
    if subset not in ["valid", "test"]:
        raise ValueError(f"Invalid subset string provided = {subset}")

    if stations is None:
        stations = Station.COORDS
    else:
        stations = {k: Station.COORDS[k] for k in stations.keys()}

    timesteps = len(previous_time_offsets)

    def data_generator():
        # Directly use the provided list of datetimes
        # We can't delete any img/target at valid/test time since they have to be in order for evaluator.py
        for batch_idx in range(0, len(target_datetimes) // batch_size + 1):
            batch_of_datetimes = target_datetimes[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if not batch_of_datetimes:
                continue

            samples, _, min_idx, max_idx = get_hdf5_samples_list_datetime(
                dataframe, batch_of_datetimes, get_previous_time_offsets(), test_time=True, patch_size=patch_size,
                directory=data_directory, stations=stations,
            )

            samples = tf.cast(tf.convert_to_tensor(samples), dtype=tf.float32)

            if normalize_imgs:
                # Normalize here since we normalize at load time for tfrecords at train time
                # This code is nasty but necessary since the normalization in extract_tf_records.py
                # had some issues, so we have to replicate the same exact (buggy) steps to get
                # accurate results on the test set
                min_idx = tf.cast(tf.convert_to_tensor(min_idx), dtype=tf.float32)
                max_idx = tf.cast(tf.convert_to_tensor(max_idx), dtype=tf.float32)

                samples = tf.dtypes.cast(((samples - min_idx) / (max_idx - min_idx)) * 255, tf.uint8)
                samples = ((tf.dtypes.cast(samples, tf.float32) / 255) * (max_idx - min_idx) + min_idx)

                samples = HDF5File.min_max_normalization_min1_1(samples)

            samples = samples[:, -timesteps:, :, :, :]

            past_metadata, future_metadata = get_metadata(
                dataframe, batch_of_datetimes, previous_time_offsets, target_time_offsets, stations
            )

            if subset == "test":
                targets = tf.zeros(shape=(len(batch_of_datetimes) * len(stations), len(target_time_offsets)))
            else:  # validation
                targets, _ = get_labels_list_datetime(
                    dataframe, batch_of_datetimes, target_time_offsets, stations
                )
            yield (samples, past_metadata, future_metadata), targets

    past_metadata_len = 5  # day, hour, min, daytime, Clearsky
    future_metadata_len = len(target_time_offsets)
    target_len = len(target_time_offsets)
    # TODO: Check if that's how prefetch should be used
    data_loader = tf.data.Dataset.from_generator(
        data_generator, output_types=((tf.float32, tf.float32, tf.float32), tf.float32),
        output_shapes=(((None, timesteps, patch_size[0], patch_size[1], 5),
                        (None, timesteps, past_metadata_len),
                        (None, future_metadata_len)),
                       (None, target_len))
    ).prefetch(tf.data.experimental.AUTOTUNE)
    return data_loader


def tfrecord_preprocess_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: List[datetime.datetime],
        target_time_offsets: List[datetime.timedelta],
        previous_time_offsets: List[datetime.timedelta],
        batch_size: int,
        data_directory: Optional[str] = None,
        patch_size: Tuple[int, int] = (32, 32),
) -> tf.data.Dataset:

    stations = Station.COORDS

    def data_generator():
        for batch_idx in range(0, len(target_datetimes) // batch_size + 1):
            batch_of_datetimes = target_datetimes[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if not batch_of_datetimes or len(batch_of_datetimes) == 0:
                continue

            samples, invalids_i, min_idx, max_idx = get_hdf5_samples_list_datetime(
                dataframe, batch_of_datetimes, previous_time_offsets, False, patch_size, data_directory, stations,
            )

            for index in sorted(invalids_i, reverse=True):
                del batch_of_datetimes[index]

            if len(batch_of_datetimes) == 0:
                continue

            past_metadata, future_metadata = get_metadata(
                dataframe, batch_of_datetimes, previous_time_offsets, target_time_offsets, stations
            )
            targets, invalid_idx_t = get_labels_list_datetime(
                dataframe, batch_of_datetimes, target_time_offsets, stations
            )

            # Remove samples and metadata at indexes of invalid targets
            for index in sorted(invalid_idx_t, reverse=True):
                del targets[index]
                del min_idx[index]
                del max_idx[index]
                del samples[index]
                del past_metadata[index]
                del future_metadata[index]

            if len(samples) == 0:
                continue

            yield (samples, past_metadata, future_metadata, min_idx, max_idx), targets

    past_metadata_len = 5  # day, hour, min, daytime, Clearsky
    future_metadata_len = len(target_time_offsets)
    timesteps = len(previous_time_offsets)
    target_len = len(target_time_offsets)
    data_loader = tf.data.Dataset.from_generator(
        data_generator, output_types=((tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
        output_shapes=(((None, timesteps, patch_size[0], patch_size[1], 5),
                        (None, timesteps, past_metadata_len),
                        (None, future_metadata_len),
                        (None, timesteps, 1, 1, 5),
                        (None, timesteps, 1, 1, 5)),
                       (None, target_len))
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return data_loader
