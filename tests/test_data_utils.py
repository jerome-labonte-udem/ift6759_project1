import unittest
import pandas as pd
from pathlib import Path
from src.utils.data_utils import (
    get_metadata_start_end, get_labels_start_end, get_labels_list_datetime, random_timestamps_from_day
)
from src.schema import Station, get_target_time_offsets
import numpy as np


class TestDataUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = Path(Path(__file__).parent.parent, "data",
                              "catalog.helios.public.20100101-20160101.pkl")
        self.test_data_path = Path("dummy_test_catalog.pkl")
        self.df = pd.read_pickle(self.data_path)
        self.datetime_hdf5_test = pd.Timestamp(np.datetime64('2015-12-31T08:00:00.000000000'))

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
        # Test that we can correctly fetch X labels from a station
        length = 96
        list_datetimes = list(self.df.index[:length].values)
        fetch_labels, invalid_indexes = get_labels_list_datetime(
            self.df, list_datetimes, get_target_time_offsets(), stations=Station.COORDS
        )
        print(self.df.head())
        # Station DRA invalid first day
        for index in invalid_indexes:
            self.assertTrue(0 < index < 100 * len(Station.list()))

    def test_random_timestamps_from_day(self):
        """ Test that starting from one day, we can randomly sample X timestamps from that day """
        bs = 10
        ts = random_timestamps_from_day(self.df, self.datetime_hdf5_test, batch_size=bs)
        self.assertEqual(10, len(ts))
