import unittest
from pathlib import Path
import pandas as pd
import datetime
from src.schema import Catalog
from src.hdf5 import HDF5File


class TestHDF5File(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = Path(Path(__file__).parent.parent, "data",
                              "catalog.helios.public.20100101-20160101.pkl")
        self.df = pd.read_pickle(self.data_path)
        # Typical target_datetimes, but we should be able to take as inputs
        # different ones according to evaluator.py
        self.target_time_offsets = [
            datetime.timedelta(hours=0),
            datetime.timedelta(hours=1),
            datetime.timedelta(hours=3),
            datetime.timedelta(hours=6)
        ]

    def test_get_hdf5_offsets(self):
        midnight = pd.Timestamp(self.df.index[0])
        offsets = HDF5File.get_offsets(midnight, self.target_time_offsets)
        self.assertEqual(16 * 4, offsets[0])
        self.assertEqual(17 * 4, offsets[1])
        self.assertEqual(19 * 4, offsets[2])
        self.assertEqual(22 * 4, offsets[3])
        # Starting at 9:45
        nine_45 = pd.Timestamp(self.df.index[39])
        offsets = HDF5File.get_offsets(nine_45, self.target_time_offsets)
        self.assertEqual(self.df.iloc[39][Catalog.hdf5_8bit_offset], offsets[0])
        self.assertEqual(7, offsets[0])
        self.assertEqual(7 + 4, offsets[1])
        self.assertEqual(7 + 4 * 3, offsets[2])
        self.assertEqual(7 + 4 * 6, offsets[3])
        # Starting at 8:00
        eight = pd.Timestamp(self.df.index[32])
        offsets = HDF5File.get_offsets(eight, self.target_time_offsets)
        self.assertEqual(self.df.iloc[32][Catalog.hdf5_8bit_offset], offsets[0])
        self.assertEqual(0, offsets[0])
        self.assertEqual(0 + 4, offsets[1])
        self.assertEqual(0 + 4 * 3, offsets[2])
        self.assertEqual(0 + 4 * 6, offsets[3])