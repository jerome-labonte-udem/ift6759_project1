from typing import List, Tuple
from collections import OrderedDict
import datetime
import pandas as pd


class Station:
    BND = "BND"
    DRA = "DRA"
    FPK = "FPK"
    GWN = "GWN"
    PSU = "PSU"
    TBL = "TBL"
    SXF = "SXF"

    # (latitude, longitude, elevation (meters))
    LATS_LONS = OrderedDict([
        (BND, (40.05192, -88.37309, 230)),
        (DRA, (36.62373, -116.01947, 1007)),
        (FPK, (48.30783, -105.1017, 634)),
        (GWN, (34.2547, -89.8729, 98)),
        (PSU, (40.72012, -77.93085, 376)),
        (TBL, (40.12498, -105.2368, 1689)),
        (SXF, (43.73403, -96.62328, 473))
    ])

    # Pre-computed coordinates of the stations in the (650, 1500) images
    COORDS = OrderedDict([
        (BND, (401, 915)),
        (DRA, (315, 224)),
        (FPK, (607, 497)),
        (GWN, (256, 878)),
        (PSU, (418, 1176)),
        (TBL, (403, 494)),
        (SXF, (493, 709))
    ])

    @staticmethod
    def list() -> List[str]:
        return [Station.BND, Station.DRA, Station.FPK, Station.GWN,
                Station.PSU, Station.TBL, Station.SXF]


class Catalog:
    ncdf_path = "ncdf_path"
    hdf5_8bit_path = "hdf5_8bit_path"
    hdf5_8bit_offset = "hdf5_8bit_offset"
    hdf5_16bit_path = "hdf5_16bit_path"
    hdf5_16_bit_offset = "hdf5_16bit_offset"
    # Shape of each channel image according to hdf5_8bit file
    size_image = (650, 1500)

    # This is a Column that we add to the DF to filter out invalid t_0s to speed up training
    is_invalid = "is_invalid"

    @staticmethod
    def clearsky_ghi(station: str) -> str:
        return f"{station}_CLEARSKY_GHI"

    @staticmethod
    def daytime(station: str) -> str:
        return f"{station}_DAYTIME"

    @staticmethod
    def ghi(station: str) -> str:
        return f"{station}_GHI"

    @staticmethod
    def cloudiness(station: str) -> str:
        return f"{station}_CLOUDINESS"

    @staticmethod
    def invalid_hours() -> List[Tuple]:
        """ Return list of invalid (hour, minute) photos """
        return [
            (0, 0), (0, 30), (3, 0), (6, 0), (9, 0),
            (12, 0), (15, 0), (15, 30), (18, 0), (21, 0)
        ]

    @staticmethod
    def add_invalid_t0_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove every possible t0 that has invalid path to hdf5 file / is an invalid hour /
        / and is night_time for all stations
        :return: same dataframe but with added column "is_invalid"
        """
        df[Catalog.is_invalid] = False

        for hour, minute in Catalog.invalid_hours():
            df[Catalog.is_invalid].mask((df.index.hour == hour) & (df.index.minute == minute), True, inplace=True)

        df[Catalog.is_invalid].mask(df[Catalog.hdf5_8bit_path] == "nan", True, inplace=True)

        df[Catalog.is_invalid].mask(
            ((df[Catalog.daytime(Station.BND)] == 0) &
             (df[Catalog.daytime(Station.DRA)] == 0) &
             (df[Catalog.daytime(Station.FPK)] == 0) &
             (df[Catalog.daytime(Station.GWN)] == 0) &
             (df[Catalog.daytime(Station.PSU)] == 0) &
             (df[Catalog.daytime(Station.TBL)] == 0) &
             (df[Catalog.daytime(Station.SXF)] == 0)), True, inplace=True
        )
        return df


def get_target_time_offsets():
    """ This format is to be compatible with evaluator.py
    We want to evaluate at t0, t0+1, t0+3, t0+6 """
    return [
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=1),
        datetime.timedelta(hours=3),
        datetime.timedelta(hours=6)
    ]


def get_previous_time_offsets():
    """Example of previous time offsets list for tests"""
    return [
        -datetime.timedelta(hours=3),
        -datetime.timedelta(hours=2, minutes=15),
        -datetime.timedelta(hours=1, minutes=30),
        -datetime.timedelta(hours=0, minutes=45),
        datetime.timedelta(hours=0)
    ]
