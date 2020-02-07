import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
from src.schema import get_target_time_offsets, Station
from src.data_pipeline import hdf5_dataloader_list_of_days
from matplotlib import pyplot as plt
import cv2 as cv


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = Path(Path(__file__).parent.parent, "data")
        self.data_path = Path(self.data_dir, "catalog.helios.public.20100101-20160101.pkl")
        self.df = pd.read_pickle(self.data_path)
        self.datetime_hdf5_last = pd.Timestamp(np.datetime64('2015-12-31T08:00:00.000000000'))
        self.datetime_hdf5_middle = pd.Timestamp(np.datetime64('2012-01-09T08:00:00.000000000'))
        self.datetime_hdf5_first = pd.Timestamp(np.datetime64('2010-01-01T08:00:00.000000000'))
        self.hdf8_dir = Path(self.data_dir, "hdf5v7_8bit")

    def test_hdf5_dataloader_list_of_days(self):
        """ Test dataloader at train/valid time (list of days) """
        # Actually using same day 3 times since there is a random sampling involved
        list_days = [self.datetime_hdf5_middle]
        dataset = hdf5_dataloader_list_of_days(
            self.df, list_days, get_target_time_offsets(), data_directory=self.hdf8_dir, patch_size=(256, 256),
            batch_size=3, test_time=False, stations={Station.GWN: Station.COORDS[Station.GWN]},
            previous_time_offsets=[datetime.timedelta(hours=-6), datetime.timedelta(hours=-3)]
        )
        norm_min = -0.009999999776482582
        norm_max = 2.252500057220459
        for (sample, metadata), target in dataset:
            self.assertEqual(len(sample), len(target))
            self.assertEqual(len(metadata), len(target))
            for b in range(3):
                for i in range(3):
                    array = np.asarray(sample[b, i, :, :, 0])
                    array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                    array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                    print(f"shape array = {array.shape}")
                    plt.imshow(array)
                    plt.show()
            for t in target:
                self.assertEqual(len(get_target_time_offsets()), len(t))

    def test_hdf5_data_loader_test_time(self):
        # Test dataloader at test time (list of target times)
        list_datetimes = [
            self.datetime_hdf5_last,
            self.datetime_hdf5_last + datetime.timedelta(hours=1),
            self.datetime_hdf5_last + datetime.timedelta(hours=3),
            self.datetime_hdf5_last + datetime.timedelta(hours=6, minutes=15),
            self.datetime_hdf5_last + datetime.timedelta(hours=12)
        ]

        dataset = hdf5_dataloader_list_of_days(self.df, list_datetimes,
                                               get_target_time_offsets(), data_directory=self.hdf8_dir,
                                               batch_size=len(list_datetimes), test_time=True)
        for (sample, metadata), target in dataset:
            self.assertEqual(len(sample), len(target))
            self.assertEqual(len(metadata), len(target))


if __name__ == '__main__':
    unittest.main()
