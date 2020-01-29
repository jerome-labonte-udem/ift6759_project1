import unittest
import pandas as pd
from pathlib import Path
from src.utils.data_utils import (
    get_metadata_start_end, get_labels_start_end, get_labels_list_datetime
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
