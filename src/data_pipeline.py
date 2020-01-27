import pdb
from src.data_utils import get_labels_list_datetime
from src.utils import fetch_hdf5_sample
from src.schema import HDF5File
from pathlib import Path
import h5py
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Tuple, AnyStr
import datetime
import random


class HDF5Dataset(tf.data.Dataset):

    def _generator(batch_size, target_datetimes, target_time_offsets, df, stations):
        """ Generate dummy data for the model, only for example purposes. """
        image_dim = (64, 64)
        # TODO: How do we choose the station ? For now random sampling
        station_str = random.choice(stations.keys())
        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i:i + batch_size]
            samples = tf.random.uniform(shape=(
                len(batch_of_datetimes), image_dim[0], image_dim[1], 5
            ))
            targets = get_labels_list_datetime(df, batch_of_datetimes, target_time_offsets, station_str)
            # Remember that you do not have access to the targets.
            # Your dataloader should handle this accordingly.
            yield samples, targets

    def __new__(cls, target_datetimes: List[datetime.datetime],
                target_time_offsets: List[datetime.timedelta],
                stations: Dict[AnyStr, Tuple[float, float, float]],
                df_metadata: pd.DataFrame, batch_size=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(batch_size, target_datetimes, target_time_offsets, df_metadata, stations)
        )


def main():
    hdf5_path = Path(Path(__file__).parent.parent,
                     "data", "hdf8", "2012.01.03.0800.h5")

    # target_channels = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    # dataframe_path = Path(Path(__file__).parent.parent,
    #                      "data", "catalog.helios.public.20100101-20160101.pkl")
    # stations = {"BND": (40.05192, -88.37309, 230), "TBL": (40.12498, -105.23680, 1689)}
    # viz_hdf5_imagery(hdf5_path, target_channels, dataframe_path, stations)
    hdf5_offset = 32

    with h5py.File(hdf5_path, "r") as h5_data:
        for channel in ["ch1", "ch2", "ch3", "ch4", "ch6"]:
            ch_data = fetch_hdf5_sample(channel, h5_data, hdf5_offset)
            print(ch_data.shape)  # channel data is saved as 2D array (HxW))

        print(f"start_time {h5_data.attrs[HDF5File.start_time]}")
        print(f"end_time {h5_data.attrs[HDF5File.end_time]}")
        lats = fetch_hdf5_sample("lat", h5_data, hdf5_offset)
        lons = fetch_hdf5_sample("lon", h5_data, hdf5_offset)
        pdb.set_trace()
        print(f"lats {lats.shape} and longs {lons.shape}")
        # [print(dataset) for dataset in h5_data]


if __name__ == "__main__":
    main()
