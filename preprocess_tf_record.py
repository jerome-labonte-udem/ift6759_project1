import pandas as pd
import argparse
import json
import os
import tqdm
import tensorflow as tf
import numpy as np
import random
from typing import Tuple, List

from src.data_pipeline import tfrecord_preprocess_dataloader
from src.schema import Catalog, get_target_time_offsets, get_previous_time_offsets

FILE_TF_RECORD = "input_file_month.tfrecord"


def shard_dataset(path_save: str):
    path_tf_record = os.path.join(path_save, FILE_TF_RECORD)
    assert os.path.exists(path_tf_record), f"Path to tf_record doesn't exist {path_tf_record}"
    size_file = os.path.getsize(path_tf_record) / (1024 ** 2)
    n_shards = max(int(size_file // 100), 1)
    print(f"Splitting tfrecord of size {size_file / 1024:.2f}GB into {n_shards} shards")
    raw_dataset = tf.data.TFRecordDataset(path_tf_record)
    for i in range(n_shards):
        writer = tf.data.experimental.TFRecordWriter(os.path.join(path_save, f"output_file-part-{i}.tfrecord"))
        writer.write(raw_dataset.shard(n_shards, i))
    # Delete original (non-sharded record)
    os.remove(path_tf_record)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_example(sample, past_metadata, future_metadata, min_channels, max_channels, target):
    feature = {
        'sample': _bytes_feature(tf.io.serialize_tensor(sample)),
        'past_metadata': _bytes_feature(tf.io.serialize_tensor(past_metadata)),
        'future_metadata': _bytes_feature(tf.io.serialize_tensor(future_metadata)),
        'min_channels': _bytes_feature(tf.io.serialize_tensor(min_channels)),
        'max_channels': _bytes_feature(tf.io.serialize_tensor(max_channels)),
        'target': _bytes_feature(tf.io.serialize_tensor(target)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def preprocess_tfrecords(
        dataframe_path: str, data_path: str, path_save: str, patch_size: Tuple[int, int],
        test_local: bool, is_validation: bool, year_month_day: Tuple[int, int, int] = None,
        list_datetimes: List = None
):
    """
    Preprocess train set / validation set in shuffled tf_records of around 100MB each
    :param dataframe_path: path to catalog.pkl
    :param data_path: path to hdf5 files
    :param path_save: directory to save, which will be further divided in /train or /valid
    :param patch_size: patch_size to store, we can further crop into it at load time
    :param test_local:
    :param is_validation: preprocess validation set (2015)
    :param year_month_day: specify (year, month, day) of target_datetimes, made for debugging !
    :return:
    """
    if is_validation:
        path_save = os.path.join(path_save, "validation")
    else:
        path_save = os.path.join(path_save, "train")
    os.makedirs(path_save, exist_ok=True)

    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    df = Catalog.add_invalid_t0_column(pd.read_pickle(dataframe_path))
    assert os.path.isdir(data_path), f"invalid data path: {data_path}"

    if list_datetimes is None:
        if is_validation:
            df = df.loc[df.index.year == 2015]
            if test_local:
                df = df.loc[df.index.month == 1]
                df = df.loc[df.index.week == 1]
        else:
            df = df.loc[df.index.year < 2015]
            if test_local:
                df = df.loc[df.index.year == 2012]
                df = df.loc[df.index.month == 1]
                # df = df.loc[df.index.day == 9]
        if year_month_day is not None and isinstance(year_month_day, tuple):
            print("Debugging preprocessing TF Record")
            df = df.loc[df.index.year == year_month_day[0]]
            df = df.loc[df.index.month == year_month_day[1]]
            df = df.loc[df.index.day == year_month_day[2]]

        target_datetimes = list(df.loc[~df[Catalog.is_invalid]].index.values)
        random.shuffle(target_datetimes)
        if len(df) == 0 or len(target_datetimes) == 0:
            raise ValueError(f"DF and target_datetimes is empty: wrong set of arguments: "
                             f"validation={is_validation}; test_local={test_local}")
    else:
        target_datetimes = list_datetimes

    print(f"Total of {len(target_datetimes)} target_datetimes to process")
    batch_size = 16

    previous_time_offsets = get_previous_time_offsets()

    dataset = tfrecord_preprocess_dataloader(
        df, target_datetimes, get_target_time_offsets(), previous_time_offsets,
        batch_size=batch_size, data_directory=data_path, patch_size=patch_size
    )
    path_tf_record = os.path.join(path_save, FILE_TF_RECORD)

    with tf.io.TFRecordWriter(path_tf_record) as writer:
        for (samples, past_metadata, future_metadata, mins, maxs), target in tqdm.tqdm(
                dataset, desc="Preparing minibatches", total=len(target_datetimes) // batch_size
        ):
            # Verify
            assert len(samples) == len(past_metadata)
            assert len(samples) == len(future_metadata)
            assert len(samples) == len(mins)
            assert len(samples) == len(maxs)
            assert len(samples) == len(target)

            # Write to TFRecord
            for sample, p, f, min_idx, max_idx, t in zip(samples, past_metadata, future_metadata, mins, maxs, target):
                if tf.reduce_any(tf.math.is_nan(sample)) or tf.reduce_any(tf.math.is_nan(p)) \
                        or tf.reduce_any(tf.math.is_nan(f)) or tf.reduce_any(tf.math.is_nan(t)):
                    continue
                sample = sample.numpy()
                min_idx, max_idx = min_idx.numpy(), max_idx.numpy()
                sample = (((sample.astype(np.float32) - min_idx) / (max_idx - min_idx)) * 255).astype(np.uint8)
                example = _convert_example(sample, p, f, min_idx, max_idx, t)
                writer.write(example)

    # Split dataset in n_shards
    shard_dataset(path_save)
    print("Done !")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path", type=str, help="path to config file"
    )
    parser.add_argument(
        "--validation", dest='validation', help="preprocess validation set", action='store_true'
    )
    parser.add_argument(
        '--test_local', dest='test_local', action='store_true'
    )
    parser.add_argument(
        '--only_shard', dest='only_shard', action='store_true'
    )
    parser.set_defaults(validation=False)
    parser.set_defaults(test_local=False)
    parser.set_defaults(only_shard=False)

    args = parser.parse_args()

    assert os.path.isfile(args.cfg_path), f"invalid config file: {args.cfg_path}"
    with open(args.cfg_path, "r") as config_file:
        config = json.load(config_file)

    if args.only_shard:
        # If the code bug/stop during sharding, run this
        if args.validation:
            shard_dataset(os.path.join(config["save_path"], "validation"))
        else:
            shard_dataset(os.path.join(config["save_path"], "train"))
    else:
        preprocess_tfrecords(
            config["dataframe_path"],
            config["data_path"],
            config["save_path"],
            (config["patch_size"], config["patch_size"]),
            args.test_local,
            args.validation
        )


if __name__ == "__main__":
    main()
