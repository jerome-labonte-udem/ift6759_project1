import tensorflow as tf
import os
import glob
from src.hdf5 import HDF5File

SAMPLE = "sample"
PAST_METADATA = "past_metadata"
FUTURE_METADATA = "future_metadata"
TARGET = "target"
MIN_CHANNELS = "min_channels"
MAX_CHANNELS = "max_channels"
INIT_PS = 64
HALF_INIT_PS = 64 // 2


def parse_dataset(dir_shards: str, cnn_2d: bool, patch_size: int):
    """
    Based on https://www.tensorflow.org/tutorials/load_data/tfrecord
    Extract our dataset from a directory of tf_records
    """
    all_files = []
    for file in glob.glob(os.path.join(dir_shards, "*.tfrecord")):
        all_files.append(file)

    raw_image_dataset = tf.data.TFRecordDataset(all_files)

    # Create a dictionary describing the features.
    feature_description = {
        SAMPLE: tf.io.FixedLenFeature([], tf.string),
        PAST_METADATA: tf.io.FixedLenFeature([], tf.string),
        FUTURE_METADATA: tf.io.FixedLenFeature([], tf.string),
        MIN_CHANNELS: tf.io.FixedLenFeature([], tf.string),
        MAX_CHANNELS: tf.io.FixedLenFeature([], tf.string),
        TARGET: tf.io.FixedLenFeature([], tf.string),
    }

    half_ps = patch_size // 2

    def _parse_image_function(serialized_example):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, feature_description)
        # convert image back to float32 and normalize in [-1, 1] range
        min_channel = tf.io.parse_tensor(example[MIN_CHANNELS], out_type=tf.float32)
        max_channel = tf.io.parse_tensor(example[MAX_CHANNELS], out_type=tf.float32)
        sample = tf.io.parse_tensor(example[SAMPLE], out_type=tf.uint8)
        sample = ((tf.dtypes.cast(sample, tf.float32) / 255) * (max_channel - min_channel) + min_channel)
        sample = HDF5File.min_max_normalization_min1_1(sample)

        past_metadata = tf.io.parse_tensor(example[PAST_METADATA], out_type=tf.float32)
        if cnn_2d:
            # take only t0 image
            sample = tf.reshape(sample[-1, :, :, :], (1, INIT_PS, INIT_PS, 5))
            past_metadata = tf.reshape(past_metadata[-1, :], (1, 5))
        else:
            sample = tf.reshape(sample, (5, INIT_PS, INIT_PS, 5))
            past_metadata = tf.reshape(past_metadata, (5, 5))

        if patch_size < INIT_PS:
            sample = sample[:, HALF_INIT_PS - half_ps:HALF_INIT_PS + half_ps,
                            HALF_INIT_PS - half_ps:HALF_INIT_PS + half_ps, :]

        future_metadata = tf.io.parse_tensor(example[FUTURE_METADATA], out_type=tf.float32)
        future_metadata = tf.reshape(future_metadata, (4,))
        target = tf.io.parse_tensor(example[TARGET], out_type=tf.float32)
        target = tf.reshape(target, (4,))

        inputs = (sample, past_metadata, future_metadata)

        return inputs, target

    return raw_image_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def filter_fn(inputs, target):
    # remove all samples that have NaN values
    cond = tf.reduce_any(tf.math.is_nan(inputs[1])) or tf.reduce_any(tf.math.is_nan(inputs[2])) \
           or tf.reduce_any(tf.math.is_nan(target))
    return not cond


def tfrecord_dataloader(
        dir_shards: str,
        cnn_2d: bool,
        patch_size: int
) -> tf.data.Dataset:
    """
    Dataloader at train time, fetch pre-shuffled batches of target_datetimes
    :param dir_shards: directory where the shards of .tfrecords are
    :param cnn_2d: if cnn2d, only return image at t0, else return [previous_images, t0]
    :param patch_size: patch_size to crop, needs to be < INIT_PS
    :return:
    """
    data_loader = parse_dataset(dir_shards, cnn_2d, patch_size)
    data_loader = data_loader.filter(filter_fn)
    data_loader.prefetch(tf.data.experimental.AUTOTUNE)
    return data_loader
