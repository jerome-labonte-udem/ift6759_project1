
class Station:
    BND = "BND"
    DRA = "DRA"
    FPK = "FPK"
    GWN = "GWN"
    PSU = "PSU"
    TBL = "TBL"
    SXF = "SXF"


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
