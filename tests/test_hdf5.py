import unittest
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from src.schema import Catalog, Station, get_target_time_offsets
from src.hdf5 import HDF5File
from src.data_pipeline import hdf5_dataloader_list_of_days


class TestHDF5File(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = Path(Path(__file__).parent.parent, "data")
        self.data_path = Path(self.data_dir, "catalog.helios.public.20100101-20160101.pkl")
        self.df = pd.read_pickle(self.data_path)
        self.datetime_hdf5_test = pd.Timestamp(np.datetime64('2012-01-03T08:00:00.000000000'))
        self.hdf8_dir = Path(self.data_dir, "hdf5v7_8bit")
        self.hdf5_path = Path(self.hdf8_dir, "2012.01.03.0800.h5")

    def test_get_hdf5_offsets(self):
        midnight = pd.Timestamp(self.df.index[0])
        offsets = HDF5File.get_offsets(midnight, get_target_time_offsets())
        self.assertEqual(16 * 4, offsets[0])
        self.assertEqual(17 * 4, offsets[1])
        self.assertEqual(19 * 4, offsets[2])
        self.assertEqual(22 * 4, offsets[3])
        # Starting at 9:45
        nine_45 = pd.Timestamp(self.df.index[39])
        offsets = HDF5File.get_offsets(nine_45, get_target_time_offsets())
        self.assertEqual(self.df.iloc[39][Catalog.hdf5_8bit_offset], offsets[0])
        self.assertEqual(7, offsets[0])
        self.assertEqual(7 + 4, offsets[1])
        self.assertEqual(7 + 4 * 3, offsets[2])
        self.assertEqual(7 + 4 * 6, offsets[3])
        # Starting at 8:00
        eight = pd.Timestamp(self.df.index[32])
        offsets = HDF5File.get_offsets(eight, get_target_time_offsets())
        self.assertEqual(self.df.iloc[32][Catalog.hdf5_8bit_offset], offsets[0])
        self.assertEqual(0, offsets[0])
        self.assertEqual(0 + 4, offsets[1])
        self.assertEqual(0 + 4 * 3, offsets[2])
        self.assertEqual(0 + 4 * 6, offsets[3])

    def test_extract_patches(self):
        """ Test to extract a (x,y) patch from a hdf5 file given an index. """
        sample_idx = 0
        patch_size = (16, 16)
        with h5py.File(self.hdf5_path, "r") as f_h5_data:
            h5 = HDF5File(f_h5_data)
            patches = h5.get_image_patches(sample_idx, Station.COORDS, patch_size=patch_size)
            for patch in patches:
                self.assertEqual((patch_size[0], patch_size[1], len(h5.CHANNELS)), patch.shape)

    def test_lats_lons_constant(self):
        """ This test was to verify if the lats/lons are all the same
        for the whole hdf5 file. They are indeed equal, meaning that we can only
        take the first one that is valid """
        with h5py.File(self.hdf5_path, "r") as f_h5_data:
            h5 = HDF5File(f_h5_data)
            lats, lons = h5.fetch_lat_long(0)
            for idx in range(h5.archive_lut_size):
                new_lats, new_lons = h5.fetch_lat_long(idx)
                if new_lats is None or new_lons is None:
                    # print(f"No lats/longs for idx {idx}")
                    continue
                np.testing.assert_equal(lats, new_lats)

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

    def test_get_stations_coords(self):
        """ Test to generate station coordinates (in pixel) on the (650, 1500) images """
        with h5py.File(self.hdf5_path, "r") as f_h5_data:
            h5 = HDF5File(f_h5_data)
            lats, lons = h5.fetch_lat_long(0)
            stations_coords = h5.get_stations_coordinates(lats, lons, Station.LATS_LONS)
            for k, (x, y) in stations_coords.items():
                print(f"{k} = ({x}, {y})")
                self.assertTrue(0 <= x < Catalog.size_image[0])
                self.assertTrue(0 <= y < Catalog.size_image[1])
