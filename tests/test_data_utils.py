import unittest
import pandas as pd
from pathlib import Path
from src.data_utils import get_metadata, get_labels
from src.schema import Station


class TestCatalog(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = Path(Path(__file__).parent.parent, "data", "catalog.helios.public.20100101-20160101.pkl")
        self.test_data_path = Path("dummy_test_catalog.pkl")

    def test_get_metadata(self):
        df = pd.read_pickle(self.data_path)
        # If end < start, return nothing
        meta = get_metadata(df, Station.BND, "2011-01-01", "2010-01-01")
        self.assertEqual(0, len(meta))
        # one day, we get 18 hours * 4 = 72 data points
        meta = get_metadata(df, Station.BND, "2010-01-01", "2010-01-02")
        self.assertEqual(72, len(meta))

    def test_get_labels(self):
        df = pd.read_pickle(self.data_path)
        # If end < start, return nothing
        labels = get_labels(df, Station.SXF, "2011-01-01", "2010-01-01")
        self.assertEqual(0, len(labels))
        # 1 day
        labels = get_labels(df, Station.SXF, "2011-01-01", "2011-01-02")
        self.assertEqual(72, len(labels))
        # 1 week
        labels = get_labels(df, Station.SXF, "2011-01-01", "2011-01-08")
        self.assertEqual(24 * 4 * 6 + 18 * 4, len(labels))
