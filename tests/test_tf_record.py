import unittest
from pathlib import Path
import os
import shutil
import tensorflow as tf
import numpy as np

from src.hdf5 import HDF5File
from preprocess_tf_record import preprocess_tfrecords
from src.extract_tf_record import tfrecord_dataloader


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
        loader = tfrecord_dataloader(records_dir, cnn_2d=False, patch_size=self.patch_size[0])
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


if __name__ == '__main__':
    unittest.main()
