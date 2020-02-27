import unittest
from pathlib import Path
import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

from src.hdf5 import HDF5File
from preprocess_tf_record import preprocess_tfrecords
from src.extract_tf_record import tfrecord_dataloader
from src.data_pipeline import hdf5_dataloader_test
from src.schema import get_target_time_offsets


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = Path(Path(__file__).parent.parent, "data")
        self.df_path = Path(self.data_dir, "catalog.helios.public.20100101-20160101.pkl")
        self.hdf8_dir = Path(self.data_dir, "hdf5v7_8bit")
        self.path_save = Path(Path(__file__).parent, "tmp")
        if os.path.isdir(self.path_save):
            shutil.rmtree(self.path_save)
        os.makedirs(self.path_save, exist_ok=True)

        self.patch_size = (64, 64)

    def test_compress_decompress(self):
        """
        This function was just made to understand/test the compression scheme for tfrecord
        to be sure nothing was messed up
        """
        prev_sample = np.array([
            [[-1.1274498e-03, 2.6733908e+02, 2.4406729e+02, 2.6804211e+02, 2.5464401e+02],
             [-9.9999998e-03, 2.6390274e+02, 2.4187447e+02, 2.6491623e+02, 2.5111459e+02]],
            [[-9.9999998e-03,  2.6658194e+02,  2.4187447e+02,  2.6759216e+02, 2.5363835e+02],
             [-9.9999998e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
        ])

        compress_sample = (((prev_sample.astype(np.float32) - HDF5File.MIN_CHANNELS)
                            / (HDF5File.MAX_CHANNELS - HDF5File.MIN_CHANNELS)) * 255).astype(np.uint8)

        after_sample = ((tf.dtypes.cast(compress_sample, tf.float32) / 255)
                        * (HDF5File.MAX_CHANNELS - HDF5File.MIN_CHANNELS) + HDF5File.MIN_CHANNELS)

        abs_diff = np.abs(prev_sample - after_sample)
        # make sure dont loose more than 1 percent of initial
        for i in range(len(HDF5File.CHANNELS)):
            max_loss = np.max(abs_diff[:, :, :, i])
            relative_pct = 100 * max_loss / HDF5File.MAX_CHANNELS[:, :, :, i]
            self.assertTrue(relative_pct < 1)

    def test_preprocess_on_empty_day(self):
        # January 2010 has lot of NaNs all day
        # Make sure preprocessing is robust to that
        preprocess_tfrecords(
            self.df_path,
            self.hdf8_dir,
            self.path_save,
            patch_size=self.patch_size,
            test_local=False,
            is_validation=False,
            year_month_day=(2010, 1, 1)
        )
        # delete temporary directory
        shutil.rmtree(self.path_save)

    def test_preprocess_and_extract(self):
        # Preprocess one day
        preprocess_tfrecords(
            self.df_path,
            self.hdf8_dir,
            self.path_save,
            patch_size=self.patch_size,
            test_local=False,
            is_validation=False,
            year_month_day=(2012, 1, 9)
        )

        self.assertTrue(os.path.isdir(self.path_save))
        records_dir = os.path.join(self.path_save, "train")
        self.assertTrue(os.path.isdir(records_dir))

        # Extract tf_record
        loader = tfrecord_dataloader(records_dir, patch_size=self.patch_size[0], seq_len=5)
        for i, ((sample, past_metadata, future_metadata), target) in enumerate(loader.batch(16)):
            tf.debugging.assert_all_finite(sample, f"sample has NaN")
            tf.debugging.assert_all_finite(past_metadata, f"past_metadata has NaN")
            tf.debugging.assert_all_finite(future_metadata, f"future_metadata has NaN")
            tf.debugging.assert_all_finite(target, f"target has NaN")
            self.assertTrue(len(sample) == len(past_metadata))
            self.assertTrue(len(sample) == len(future_metadata))
            self.assertTrue(len(sample) == len(target))
            # make sure they are normalized
            self.assertTrue(tf.reduce_max(sample) <= 1.01)
            self.assertTrue(tf.reduce_min(sample) >= -1.01)

        # delete temporary directory
        shutil.rmtree(self.path_save)

    def test_compare_hdf5_and_tfrecord_loader(self):
        """
        Make sure the output of hdf5 dataloader and tfrecord dataloader are exactly the same
        """

        list_datetimes_str = [
            "2012-01-08T14:15:00",
            "2012-01-08T19:15:00",
            "2012-01-08T22:15:00",
            "2012-01-09T08:30:00",
            "2012-01-09T12:30:00",
            "2012-01-09T17:45:00",
            "2015-12-31T15:30:00",
            "2015-12-31T22:45:00",
            "2015-12-31T19:45:00",
        ]

        list_datetimes_datetime = [datetime.datetime.fromisoformat(d) for d in list_datetimes_str]

        previous_time_offsets = [
            -datetime.timedelta(hours=1, minutes=30),
            -datetime.timedelta(hours=0, minutes=45),
            datetime.timedelta(hours=0)
        ]

        preprocess_tfrecords(
            self.df_path,
            self.hdf8_dir,
            self.path_save,
            patch_size=self.patch_size,
            test_local=False,
            is_validation=False,
            year_month_day=None,
            list_datetimes=list_datetimes_str
        )

        self.assertTrue(os.path.isdir(self.path_save))
        records_dir = os.path.join(self.path_save, "train")
        self.assertTrue(os.path.isdir(records_dir))

        tfrecord_loader = tfrecord_dataloader(records_dir, patch_size=self.patch_size[0],
                                              seq_len=len(previous_time_offsets))
        df = pd.read_pickle(self.df_path)
        hdf5_loader = hdf5_dataloader_test(
            df, list_datetimes_datetime, get_target_time_offsets(), previous_time_offsets,
            batch_size=1, subset="valid", data_directory=self.hdf8_dir, patch_size=self.patch_size,
            normalize_imgs=True
        )

        for ((tf_sample, tf_pmd, tf_fmd), tf_target), ((hd_sample, hd_pmd, hd_fmd), hd_target) \
                in zip(tfrecord_loader.batch(7), hdf5_loader):
            tf.assert_equal(tf_target, hd_target)
            tf.assert_equal(tf_pmd, hd_pmd)
            tf.assert_equal(tf_fmd, hd_fmd)
            ratio = tf_sample / hd_sample
            print(f"Min ratio = {tf.reduce_min(ratio)} -- Max ratio = {tf.reduce_max(ratio)}")
            self.assertTrue(tf.reduce_min(ratio) > 0.975)
            self.assertTrue(tf.reduce_max(ratio) < 1.025)


if __name__ == '__main__':
    unittest.main()
