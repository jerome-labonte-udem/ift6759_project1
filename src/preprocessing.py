import pandas as pd
from pathlib import Path
import os
import tqdm
from typing import Optional
import h5py
import numpy as np
from src.hdf5 import HDF5File
from src.schema import Catalog


def calculate_max_min_per_channel(df: pd.DataFrame, directory: Optional[str] = None):
    # 2015 is validation set
    df = df.loc[df.index.year < 2015]
    df_days = df.groupby([Catalog.hdf5_8bit_path])
    maxs = [0, 0, 0, 0, 0]
    mins = [0, 0, 0, 0, 0]

    list_paths = df_days.groups.keys()

    for path in tqdm.tqdm(list_paths, desc="Computing Mins/Maxs accross channels"):
        if directory is None:
            hdf5_path = path
        else:
            folder, filename = os.path.split(path)
            hdf5_path = os.path.join(directory, filename)

        with h5py.File(hdf5_path, "r") as f_h5:
            h5 = HDF5File(f_h5)
            for i, channel in enumerate(HDF5File.CHANNELS):
                _min, _max = h5.orig_min(channel), h5.orig_max(channel)
                if _max is not None and not np.isnan(_max):
                    maxs[i] = max(maxs[i], _max)
                if _min is not None and not np.isnan(_min):
                    mins[i] = min(mins[i], _min)

    print(f"maxs = {maxs}")
    print(f"mins = {mins}")


def main():
    data_dir = Path(Path(__file__).parent.parent, "data")
    data_path = Path(data_dir, "catalog.helios.public.20100101-20160101.pkl")
    local_dir = os.path.join(data_dir, "hdf5v7_8bit")
    df = Catalog.add_invalid_t0_column(pd.read_pickle(data_path))
    calculate_max_min_per_channel(df, None)


if __name__ == "__main__":
    main()
