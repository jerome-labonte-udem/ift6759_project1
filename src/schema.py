from typing import List
from collections import OrderedDict
import datetime


class Station:
    BND = "BND"
    DRA = "DRA"
    FPK = "FPK"
    GWN = "GWN"
    PSU = "PSU"
    TBL = "TBL"
    SXF = "SXF"

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


def get_target_time_offsets():
    """ This format is to be compatible with evaluator.py
    We want to evaluate at t0, t0+1, t0+3, t0+6 """
    return [
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=1),
        datetime.timedelta(hours=3),
        datetime.timedelta(hours=6)
    ]
