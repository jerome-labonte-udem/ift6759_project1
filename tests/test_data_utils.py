import unittest
import pandas as pd
from pathlib import Path
from src.utils.data_utils import (
    get_metadata_start_end, get_labels_start_end, get_labels_list_datetime, random_timestamps_from_day,
    filter_catalog
)
from src.schema import Station
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
        self.datetime_hdf5_test = pd.Timestamp(np.datetime64('2012-01-03T08:00:00.000000000'))

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
        # true_labels_t0 = np.array(list(self.df[Catalog.ghi(Station.BND)][:100].values))
        list_datetimes = list(self.df.index[:100].values)
        fetch_labels = get_labels_list_datetime(self.df, target_datetimes=list_datetimes,
                                                target_time_offsets=self.target_time_offsets,
                                                stations=Station.COORDS)
        self.assertEqual(100 * len(Station.COORDS), len(fetch_labels))

    def test_random_timestamps_from_day(self):
        """ Test that starting from one day, we can randomly sample X timestamps from that day """
        bs = 10
        ts = random_timestamps_from_day(self.df, self.datetime_hdf5_test, batch_size=bs)
        self.assertEqual(10, len(ts))

    def test_filter_catalog(self):
        len_df_bef = len(self.df)
        df = filter_catalog(self.df, remove_invalid_labels=True)
        self.assertTrue(len(df) < len_df_bef)
