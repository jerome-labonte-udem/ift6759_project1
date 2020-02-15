import pandas as pd
import argparse
import json
import os
import tqdm
import tensorflow as tf
import numpy as np
import random


from src.data_pipeline import tfrecord_dataloader
from src.schema import Catalog, get_target_time_offsets, get_previous_time_offsets

FILE_TF_RECORD = "input_file_month.tfrecord"


def shard_dataset(path_save: str, n_shards: int):
    raw_dataset = tf.data.TFRecordDataset(os.path.join(path_save, FILE_TF_RECORD))
    for i in range(n_shards):
        writer = tf.data.experimental.TFRecordWriter(os.path.join(path_save, f"output_file-part-{i}.tfrecord"))
        writer.write(raw_dataset.shard(n_shards, i))


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


def preprocess_tfrecords(path_config: str, test_local: bool, is_validation: bool):

    assert os.path.isfile(path_config), f"invalid config file: {path_config}"
    with open(path_config, "r") as config_file:
        config = json.load(config_file)

    batch_size = 16
    patch_size = (config["patch_size"], config["patch_size"])

    dataframe_path = config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    df = Catalog.add_invalid_t0_column(pd.read_pickle(dataframe_path))

    data_path = config["data_path"]
    assert os.path.isdir(data_path), f"invalid data path: {data_path}"

    path_save = config["save_path"]
    if is_validation:
        path_save = os.path.join(path_save, "validation")
        df = df.loc[df.index.year == 2015]
        if test_local:
            df = df.loc[df.index.month == 1]
    else:
        path_save = os.path.join(path_save, "train")
        df = df.loc[df.index.year < 2015]
        if test_local:
            df = df.loc[df.index.year == 2012]
            df = df.loc[df.index.month == 1]

    os.makedirs(path_save, exist_ok=True)

    target_datetimes = list(df.loc[~df[Catalog.is_invalid]].index.values)
    random.shuffle(target_datetimes)

    print(f"Total of {len(target_datetimes)} target_datetimes to process")

    previous_time_offsets = get_previous_time_offsets()

    dataset = tfrecord_dataloader(
        df, target_datetimes, get_target_time_offsets(), previous_time_offsets,
        batch_size=batch_size, data_directory=data_path, patch_size=patch_size
    )
    # approx 0.1MB - per target_datetime --> divide by 1000 to get around 100MB per shard
    n_shards = max(len(target_datetimes) // 1000, 1)
    print(f"n_shards estimated = {n_shards}")
    path_tf_record = os.path.join(path_save, FILE_TF_RECORD)

    with tf.io.TFRecordWriter(path_tf_record) as writer:
        for (samples, past_metadata, future_metadata, mins, maxs), target in tqdm.tqdm(
                dataset, desc="Preparing minibatches", total=len(target_datetimes) // batch_size
        ):
            for sample, p, f, min_idx, max_idx, t in zip(samples, past_metadata, future_metadata, mins, maxs, target):
                sample = sample.numpy()
                min_idx, max_idx = min_idx.numpy(), max_idx.numpy()
                sample = (((sample.astype(np.float32) - min_idx) / (max_idx - min_idx)) * 255).astype(np.uint8)
                example = _convert_example(sample, p, f, min_idx, max_idx, t)
                writer.write(example)

    shard_dataset(path_save, n_shards)
    # Delete original (non-sharded record)
    os.remove(path_tf_record)


if __name__ == "__main__":
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
    parser.set_defaults(validation=False)
    parser.set_defaults(test_local=False)

    args = parser.parse_args()
    preprocess_tfrecords(args.cfg_path, args.test_local, args.validation)
