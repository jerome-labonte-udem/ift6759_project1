import unittest
import pandas as pd
from pathlib import Path
from src.data_utils import (
    get_metadata_start_end, get_labels_start_end, get_labels_list_datetime, get_hdf5_offsets
)
from src.schema import Station, Catalog
import datetime
import numpy as np


class TestDataUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = Path(Path(__file__).parent.parent, "data",
                              "catalog.helios.public.20100101-20160101.pkl")
        self.test_data_path = Path("dummy_test_catalog.pkl")
        self.df = pd.read_pickle(self.data_path)
        # Typical target_datetimes, but we should be able to take as inputs
        # different ones according to evaluator.py
        self.target_time_offsets = [
            datetime.timedelta(hours=0),
            datetime.timedelta(hours=1),
            datetime.timedelta(hours=3),
            datetime.timedelta(hours=6)
        ]

    def test_get_metadata_start_end(self):
        # If end < start, return nothing
        meta = get_metadata_start_end(self.df, Station.BND, "2011-01-01", "2010-01-01")
        self.assertEqual(0, len(meta))
        # one day, we get 18 hours * 4 = 72 data points
        meta = get_metadata_start_end(self.df, Station.BND, "2010-01-01", "2010-01-02")
        self.assertEqual(72, len(meta))

    def test_get_labels_start_end(self):
        # If end < start, return nothing
        labels = get_labels_start_end(self.df, Station.SXF, "2011-01-01", "2010-01-01")
        self.assertEqual(0, len(labels))
        # 1 day
        labels = get_labels_start_end(self.df, Station.SXF, "2011-01-01", "2011-01-02")
        self.assertEqual(72, len(labels))
        # 1 week
        labels = get_labels_start_end(self.df, Station.SXF, "2011-01-01", "2011-01-08")
        self.assertEqual(24 * 4 * 6 + 18 * 4, len(labels))

    def test_get_labels_list_datetime(self):
        # Test that we can correctly fetch 100 labels from a station
        true_labels_t0 = np.array(list(self.df[Catalog.ghi(Station.BND)][:100].values))

        list_datetimes = list(self.df.index[:100].values)
        fetch_labels = get_labels_list_datetime(self.df, target_datetimes=list_datetimes,
                                                target_time_offsets=self.target_time_offsets,
                                                station=Station.BND)
        self.assertEqual(100, len(fetch_labels))

        # Make sure we fetched the right labels
        assert (true_labels_t0 == fetch_labels[:, 0]).all()

    def test_get_hdf5_offsets(self):
        midnight = pd.Timestamp(self.df.index[0])
        offsets = get_hdf5_offsets(midnight, self.target_time_offsets)
        self.assertEqual(16 * 4, offsets[0])
        self.assertEqual(17 * 4, offsets[1])
        self.assertEqual(19 * 4, offsets[2])
        self.assertEqual(22 * 4, offsets[3])
        # Starting at 9:45
        nine_45 = pd.Timestamp(self.df.index[39])
        offsets = get_hdf5_offsets(nine_45, self.target_time_offsets)
        self.assertEqual(self.df.iloc[39][Catalog.hdf5_8bit_offset], offsets[0])
        self.assertEqual(7, offsets[0])
        self.assertEqual(7 + 4, offsets[1])
        self.assertEqual(7 + 4 * 3, offsets[2])
        self.assertEqual(7 + 4 * 6, offsets[3])
        # Starting at 8:00
        eight = pd.Timestamp(self.df.index[32])
        offsets = get_hdf5_offsets(eight, self.target_time_offsets)
        self.assertEqual(self.df.iloc[32][Catalog.hdf5_8bit_offset], offsets[0])
        self.assertEqual(0, offsets[0])
        self.assertEqual(0 + 4, offsets[1])
        self.assertEqual(0 + 4 * 3, offsets[2])
        self.assertEqual(0 + 4 * 6, offsets[3])
