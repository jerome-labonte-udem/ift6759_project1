import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
from src.schema import get_target_time_offsets, Station, Catalog
from src.data_pipeline import hdf5_dataloader_list_of_days
from matplotlib import pyplot as plt
import cv2 as cv


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = Path(Path(__file__).parent.parent, "data")
        self.data_path = Path(self.data_dir, "catalog.helios.public.20100101-20160101.pkl")
        self.df = Catalog.add_invalid_t0_column(pd.read_pickle(self.data_path))
        self.datetime_hdf5_last = pd.Timestamp(np.datetime64('2015-12-31T08:00:00.000000000'))
        self.datetime_hdf5_middle_1 = pd.Timestamp(np.datetime64('2012-01-08T08:00:00.000000000'))
        self.datetime_hdf5_middle_2 = pd.Timestamp(np.datetime64('2012-01-09T08:00:00.000000000'))
        self.datetime_hdf5_first = pd.Timestamp(np.datetime64('2010-01-01T08:00:00.000000000'))
        self.hdf8_dir = Path(self.data_dir, "hdf5v7_8bit")
        self.previous_time_offsets = [
                datetime.timedelta(hours=-6), datetime.timedelta(hours=-3), datetime.timedelta(hours=-1)
        ]

    def test_hdf5_dataloader_list_of_days(self):
        """
        Test dataloader at train/valid time (list of days)
        Test 1) Previous time offsets are given (looking at past days)
        Test 2) Only looking at T0
        """
        list_days = [self.datetime_hdf5_first, self.datetime_hdf5_middle_2, self.datetime_hdf5_last]
        dataset = hdf5_dataloader_list_of_days(
            self.df, list_days, get_target_time_offsets(), data_directory=self.hdf8_dir,
            patch_size=(32, 32), batch_size=8, subset="train", previous_time_offsets=self.previous_time_offsets
        )

        for (sample, past_metadata, future_metadata), target in dataset:
            self.assertEqual(len(sample), len(target))
            self.assertEqual(len(past_metadata), len(target))
            self.assertEqual(len(future_metadata), len(target))
            for t in target:
                self.assertEqual(len(get_target_time_offsets()), len(t))

    def test_hdf5_data_loader_test_time(self):
        # Test dataloader at test time (list of target times)
        # Test with two cases, 1) we have some previous time offsets (looking at past images)
        # and 2) only looking at t0s
        list_datetimes = [
            self.datetime_hdf5_first,
            self.datetime_hdf5_first + datetime.timedelta(hours=3),
            self.datetime_hdf5_middle_2,
            self.datetime_hdf5_middle_2 + datetime.timedelta(hours=1, minutes=15),
            self.datetime_hdf5_last,
            self.datetime_hdf5_last + datetime.timedelta(hours=1),
            self.datetime_hdf5_last + datetime.timedelta(hours=3),
            self.datetime_hdf5_last + datetime.timedelta(hours=6, minutes=15),
            self.datetime_hdf5_last + datetime.timedelta(hours=12)
        ]

        dataset = hdf5_dataloader_list_of_days(
            self.df, list_datetimes, get_target_time_offsets(), data_directory=self.hdf8_dir,
            batch_size=len(list_datetimes), subset="test", previous_time_offsets=self.previous_time_offsets,
            stations=Station.COORDS
        )
        for (sample, past_metadata, future_metadata), target in dataset:
            self.assertEqual(len(sample), len(target))
            self.assertEqual(len(past_metadata), len(target))
            self.assertEquals(len(future_metadata), len(target))

    def test_visualize_one_sample(self):
        """ Visualize all photos (previous and t0) of a sample"""
        list_days = [self.datetime_hdf5_middle_2]
        norm_min = -0.009999999776482582
        norm_max = 2.252500057220459
        hours_min = [
            (-12, 0), (-9, 0), (-6, 0), (-3, 0),
            (-2, 0), (-1, 0), (0, -30), (0, -15), (0, 0)
        ]
        dataset = hdf5_dataloader_list_of_days(
            self.df, list_days, get_target_time_offsets(), data_directory=self.hdf8_dir, patch_size=(256, 256),
            batch_size=8, subset="train", stations={Station.GWN: Station.COORDS[Station.GWN]},
            previous_time_offsets=[datetime.timedelta(hours=h, minutes=m) for h, m in hours_min[:-1]]
        )

        fig, axs = plt.subplots(3, 3)
        for idx, ((sample, past_metadata, future_metadata), target) in enumerate(dataset):
            if sample.shape[1] != len(hours_min):
                continue
            for i in range(sample.shape[1]):
                array = np.asarray(sample[0, i, :, :, 0])
                array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                axs[i // 3, i % 3].imshow(array)
                axs[i // 3, i % 3].set_title(f'({hours_min[i][0]}, {hours_min[i][1]})')
            break
        plt.figure(figsize=(30, 30))
        plt.show()


if __name__ == '__main__':
    unittest.main()
