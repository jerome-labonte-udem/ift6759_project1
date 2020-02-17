import unittest
import pandas as pd
from pathlib import Path
from src.utils.data_utils import (
    get_metadata, get_labels_list_datetime
)
from src.schema import Station, Catalog, get_target_time_offsets, get_previous_time_offsets
import numpy as np


class TestDataUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = Path(Path(__file__).parent.parent, "data",
                              "catalog.helios.public.20100101-20160101.pkl")
        self.test_data_path = Path("dummy_test_catalog.pkl")
        self.df = pd.read_pickle(self.data_path)
        self.datetime_hdf5_test = pd.Timestamp(np.datetime64('2015-12-31T08:00:00.000000000'))

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

    def test_add_invalid_t0_column(self):
        print(f"Dataframe has total of {len(self.df)} rows")
        df = Catalog.add_invalid_t0_column(self.df)
        print(f"Number of invalid rows for t0 = {len(df.loc[df['is_invalid']])}")

    def test_get_metadata(self):
        start_index = 24
        length = 10
        list_datetimes = list(self.df.index[start_index:start_index+length].values)
        df = Catalog.add_invalid_t0_column(self.df)
        past_metadata, future_metadata = get_metadata(df, list_datetimes, get_previous_time_offsets(),
                                                      get_target_time_offsets(), Station.COORDS)
        self.assertEqual(len(past_metadata), length * len(Station.list()))
        self.assertEqual(len(future_metadata), length * len(Station.list()))
        self.assertEqual(len(past_metadata[0]), len(get_previous_time_offsets()))
