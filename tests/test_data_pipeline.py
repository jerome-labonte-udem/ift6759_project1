import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from src.schema import get_target_time_offsets
from src.data_pipeline import hdf5_dataloader_list_of_days


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = Path(Path(__file__).parent.parent, "data")
        self.data_path = Path(self.data_dir, "catalog.helios.public.20100101-20160101.pkl")
        self.df = pd.read_pickle(self.data_path)
        self.datetime_hdf5_test = pd.Timestamp(np.datetime64('2012-01-03T08:00:00.000000000'))
        self.hdf8_dir = Path(self.data_dir, "hdf5v7_8bit")

    def test_hdf5_dataloader_list_of_days(self):
        """ Test dataloader  """
        batch_size = 4
        # Actually using same day 3 times since there is a random sampling involved
        list_days = [self.datetime_hdf5_test] * 3
        dataset = hdf5_dataloader_list_of_days(self.df, list_days,
                                               get_target_time_offsets(), data_directory=self.hdf8_dir,
                                               batch_size=batch_size, test_time=False)
        for (sample, metadata), target in dataset:
            self.assertEqual(len(sample), len(target))
            self.assertEqual(len(metadata), len(target))
            for t in target:
                self.assertEqual(len(get_target_time_offsets()), len(t))

        # Test dataloader at test time
        dataset = hdf5_dataloader_list_of_days(self.df, list_days,
                                               get_target_time_offsets(), data_directory=self.hdf8_dir,
                                               batch_size=batch_size, test_time=True)
        for (sample, metadata), target in dataset:
            self.assertEqual(len(sample), len(target))
            self.assertEqual(len(metadata), len(target))


if __name__ == '__main__':
    unittest.main()
