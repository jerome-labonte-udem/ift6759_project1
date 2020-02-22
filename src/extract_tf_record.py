import tensorflow as tf
import os
import glob
from typing import Optional
from src.hdf5 import HDF5File

SAMPLE = "sample"
PAST_METADATA = "past_metadata"
FUTURE_METADATA = "future_metadata"
TARGET = "target"
MIN_CHANNELS = "min_channels"
MAX_CHANNELS = "max_channels"
INIT_PS = 64
HALF_INIT_PS = 64 // 2


def parse_dataset(dir_shards: str, patch_size: int, seq_len: Optional[int]):
    """
    Based on https://www.tensorflow.org/tutorials/load_data/tfrecord
    Extract our dataset from a directory of tf_records
    """
    if seq_len < 1 or seq_len > 5:
        raise ValueError(f"Sequence length has to be between 1 and 5: Provided = {seq_len}")

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

        sample = tf.reshape(sample[-seq_len:, :, :, :], (seq_len, INIT_PS, INIT_PS, 5))
        past_metadata = tf.reshape(past_metadata[-seq_len:, :], (seq_len, 5))

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


class DataAugmentation:
    def __init__(self, prob_drop_imgs: float):
        self.prob_drop_imgs = prob_drop_imgs

    @staticmethod
    def rotate_tensor(inputs, label):
        """
        Data augmentation to randomly rotate array of past images + t0
        to either [0, 90, 180, 270] degrees
        """
        sample, past_metadata, future_metadata = inputs
        n_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        sample = tf.image.rot90(sample, n_rotations)
        return (sample, past_metadata, future_metadata), label

    def drop_imgs(self, inputs, label):
        sample, past_metadata, future_metadata = inputs
        p_sampled = tf.random.uniform(shape=[len(sample) - 1], dtype=tf.float32)
        sample = tf.unstack(sample)
        # Never remove t0 since we always have it at test time
        for i in range(len(sample) - 1):
            sample[i] = tf.cond(
                p_sampled[i] < self.prob_drop_imgs,
                lambda: tf.fill(sample[i].shape, -1.0),
                lambda: sample[i]
            )
        sample = tf.stack(sample)
        return (sample, past_metadata, future_metadata), label


def filter_fn(inputs, target):
    # remove all samples that have NaN values
    cond = tf.reduce_any(tf.math.is_nan(inputs[1])) or tf.reduce_any(tf.math.is_nan(inputs[2])) \
           or tf.reduce_any(tf.math.is_nan(target))
    return not cond


def tfrecord_dataloader(
        dir_shards: str,
        patch_size: int,
        seq_len: int,
        rotate_imgs: bool = False,
        prob_drop_imgs: float = 0.,
) -> tf.data.Dataset:
    """
    Dataloader at train time, fetch pre-shuffled batches of target_datetimes
    :param rotate_imgs: rotate images to [0, 90, 180, or 270] degrees
    :param dir_shards: directory where the shards of .tfrecords are
    :param patch_size: patch_size to crop, needs to be < INIT_PS
    :param seq_len: maximum sequence length, will return whole sequence if None
    :param prob_drop_imgs: probability to randomly drop past imags
    :return:
    """
    data_loader = parse_dataset(dir_shards, patch_size, seq_len)
    data_loader = data_loader.filter(filter_fn)

    data_aug = DataAugmentation(prob_drop_imgs)
    if rotate_imgs:
        print(f"Data augmentation: Rotating images")
        data_loader = data_loader.map(data_aug.rotate_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if seq_len > 1 and isinstance(prob_drop_imgs, float) and prob_drop_imgs > 0:
        print(f"Data Augmentation: Randomly dropping past images with probability = {prob_drop_imgs}")
        data_loader = data_loader.map(data_aug.drop_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    data_loader = data_loader.prefetch(tf.data.experimental.AUTOTUNE)
    return data_loader
