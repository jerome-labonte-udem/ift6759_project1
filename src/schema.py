from typing import List


class Station:
    BND = "BND"
    DRA = "DRA"
    FPK = "FPK"
    GWN = "GWN"
    PSU = "PSU"
    TBL = "TBL"
    SXF = "SXF"

    COORDS = {
        BND: (40.05192, -88.37309, 230),
        DRA: (36.62373, -116.01947, 1007),
        FPK: (48.30783, -105.1017, 634),
        GWN: (34.2547, -89.8729, 98),
        PSU: (40.72012, -77.93085, 376),
        TBL: (40.12498, -105.2368, 1689),
        SXF: (43.73403, -96.62328, 473)
    }

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

    @staticmethod
    def clearsky_ghi(station: str) -> str:
        return f"{station}_CLEARSKY_GHI"

    @staticmethod
    def daytime(station: str) -> str:
        return f"{station}_DAYTIME"

    @staticmethod
    def ghi(station: str) -> str:
        return f"{station}_GHI"



