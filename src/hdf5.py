import datetime
import numpy as np
from typing import List, Any, Dict, Tuple
import collections
import pandas as pd
from src.utils.utils import decompress_array
from src.schema import Catalog


class HDF5File:
    CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch6"]

    # Precomputed min/max on 2010-2014 data for all channels
    MAX_CHANNELS = np.array([3.04, 341.59998, 295.11, 341.18, 321.78]).reshape((1, 1, 1, 5))
    MIN_CHANNELS = np.array([-0.01, 0, 0, 0, 0]).reshape((1, 1, 1, 5))

    def __init__(self, hdf5_file):
        self._file = hdf5_file
        self._start_idx = self._file.attrs["global_dataframe_start_idx"]
        self._end_idx = self._file.attrs["global_dataframe_end_idx"]
        self._start_time = datetime.datetime.strptime(self._file.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
        self._end_time = self._file.attrs["global_dataframe_end_time"]

    @property
    def file(self):
        return self._file

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def archive_lut_size(self):
        # Typically == 96 (24 * 4) since we have a sample each 15 minutes
        return self.end_idx - self.start_idx

    @staticmethod
    def min_max_normalization_0_1(array: np.ndarray):
        # Min-max normalisation in the [0, 1] range
        return (array - HDF5File.MIN_CHANNELS) / (HDF5File.MAX_CHANNELS - HDF5File.MIN_CHANNELS)

    @staticmethod
    def min_max_normalization_min1_1(array):
        # Min-max normalisation in the [-1, 1] range
        return (2 * (array - HDF5File.MIN_CHANNELS) / (HDF5File.MAX_CHANNELS - HDF5File.MIN_CHANNELS)) - 1

    def lut_time_stamps(self):
        return [self.start_time + idx * datetime.timedelta(minutes=15) for idx in range(self.archive_lut_size)]

    def fetch_sample(
            self,
            dataset_name: str,
            sample_idx: int,
    ) -> Any:
        """Decodes and returns a single sample from an HDF5 dataset.
        Args:
            dataset_name: name of the HDF5 dataset to fetch the sample from using the reader. In the context of
                the GHI prediction project, this may be for example an imagery channel name (e.g. "ch1").
            sample_idx: the integer index (or offset) that corresponds to the position of the sample in the dataset.
        Returns:
            The sample. This function will automatically decompress the sample if it was compressed. It the sample is
            unavailable because the input was originally masked, the function will return ``None``. The sample itself
            may be a scalar or a numpy array.
        """
        dataset_lut_name = dataset_name + "_LUT"
        if dataset_lut_name in self.file:
            sample_idx = self.file[dataset_lut_name][sample_idx]
            if sample_idx == -1:
                return None  # unavailable
        dataset = self.file[dataset_name]
        if "compr_type" not in dataset.attrs:
            # must have been compressed directly (or as a scalar); return raw output
            return dataset[sample_idx]
        compr_type, orig_dtype, orig_shape = dataset.attrs["compr_type"], None, None
        if "orig_dtype" in dataset.attrs:
            orig_dtype = dataset.attrs["orig_dtype"]
        if "orig_shape" in dataset.attrs:
            orig_shape = dataset.attrs["orig_shape"]
        if "force_cvt_uint8" in dataset.attrs and dataset.attrs["force_cvt_uint8"]:
            array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint8, shape=orig_shape)
            orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
            array = ((array.astype(np.float32) / 255) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
        elif "force_cvt_uint16" in dataset.attrs and dataset.attrs["force_cvt_uint16"]:
            array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint16, shape=orig_shape)
            orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
            array = ((array.astype(np.float32) / 65535) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
        else:
            array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=orig_dtype, shape=orig_shape)
        return array

    def fetch_lat_long(self, sample_idx: int) -> Tuple:
        # Fetch (latitude, longitude) from file
        return self.fetch_sample("lat", sample_idx), self.fetch_sample("lon", sample_idx)

    def orig_min(self, channel_name: str):
        return self.file[channel_name].attrs.get("orig_min", None)

    def orig_max(self, channel_name: str):
        return self.file[channel_name].attrs.get("orig_max", None)

    @staticmethod
    def get_stations_coordinates(lats, lons, stations_lats_lons: Dict[str, Tuple]) -> Dict[str, Tuple]:
        """
        Taken from viz_hdf5_imagery()
        :param lats:
        :param lons:
        :param stations_lats_lons: dictionnary of str -> (latitude, longitude) of the station(s)
        :return: dictionnary of str -> (coord_x, coord_y) in the numpy array
        """
        stations_coords = {}
        for reg, lats_lons in stations_lats_lons.items():
            coords = (np.argmin(np.abs(lats - lats_lons[0])), np.argmin(np.abs(lons - lats_lons[1])))
            stations_coords[reg] = coords
        return stations_coords

    def get_image_patches(
            self,
            sample_idx: int,
            test_time: bool,
            stations_coords: collections.OrderedDict,
            patch_size: Tuple[int, int] = (16, 16),
    ) -> np.array:
        """
        :param test_time:
        :param sample_idx: index in the hdf5 file
        :param stations_coords: dictionnary of str -> (coord_x, coord_y) in the numpy array
        :param patch_size: size of the image crop that we will take
        :return: patches: List of patches where each patch has shape
        (patch_size[0], patch_size[1], N_CHANNELS)
        """
        if len(next(iter(stations_coords.values()))) != 2:
            raise ValueError(f"Invalid stations_coords, should be of len = 2, i.e. (x_coord, y_coord)")
        if patch_size[0] != patch_size[1]:
            raise NotImplementedError("Handling of non-squared patches is not implemented")
        if patch_size[0] % 2 != 0:
            raise NotImplementedError("Handling of odds patches is not implemented")

        channel_data = []
        for channel_name in self.CHANNELS:
            data = self.fetch_sample(channel_name, sample_idx)
            # Even if a channel is missing or is NaN, return an array of zeros to always predict
            # something at validation/test time
            if data is None or np.isnan(data).any():
                if test_time:  # At test time we always have to predict
                    channel_data.append(np.zeros(Catalog.size_image))
                else:
                    return None
            else:
                channel_data.append(data)

        channel_data = np.asarray(channel_data)  # transform to np.array for multidimensional slicing

        patches = []
        # Stations_coords is OrderedDict so won't mess up the order
        for coords in stations_coords.values():
            x_idx = (int(coords[0] - patch_size[0] // 2), int(coords[0] + patch_size[0] // 2))
            y_idx = (int(coords[1] - patch_size[0] // 2), int(coords[1] + patch_size[0] // 2))
            patch = channel_data[:, x_idx[0]:x_idx[1], y_idx[0]:y_idx[1]]
            patch = np.transpose(patch, (1, 2, 0))
            patches.append(patch)
        return np.array(patches)

    @staticmethod
    def get_offsets(t0: pd.Timestamp, target_time_offsets: List[datetime.timedelta]) -> List[Tuple[int, bool]]:
        """ Transform essentially a list of timestamps into hdf5_offsets to retrieve images
        e.g. hdf5_offset = 32  -> corresponds to: 2010.06.01.0800 + (32)*15min = 2010.06.01.1600
        This is also referred as sample_idx in other functions
        :return list of tuples of [offset, is_in_previous
        """
        hdf5_offsets = []
        start_day = t0.replace(hour=8, minute=0, second=0)
        for time_offset in target_time_offsets:
            time = t0 + time_offset
            # Get how many steps of 15 minutes needed to get to the hour
            ts = (time - start_day) / np.timedelta64(15, 'm')
            if ts < 0:
                #  If time is negative, i.e. before < 0800, find appropriate offset
                ts = 4 * 24 + ts
                hdf5_offsets.append((int(ts), True))
            else:
                hdf5_offsets.append((int(ts), False))
        return hdf5_offsets
