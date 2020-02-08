import pandas as pd
import numpy as np
from pathlib import Path
from src.schema import Station, Catalog


def compute_clearsky_rmse(df: pd.DataFrame):
    ghi, clearsky = [], []
    for station in Station.list():
        df_new = df.dropna(subset=[Catalog.ghi(station)])
        df_new = df_new.loc[df[Catalog.daytime(station)] == 1]
        rmse = ((df_new[Catalog.ghi(station)] - df_new[Catalog.clearsky_ghi(station)]) ** 2).mean() ** 0.5
        ghi.extend(df_new[Catalog.ghi(station)].values)
        clearsky.extend(df_new[Catalog.clearsky_ghi(station)].values)
        print(f"RMSE for station {station} = {rmse}")
    rmse_total = ((np.asarray(ghi) - np.asarray(clearsky)) ** 2).mean() ** 0.5
    print(f"Overall RMSE is {rmse_total}")


def main():
    data_dir = Path(Path(__file__).parent.parent, "data")
    data_path = Path(data_dir, "catalog.helios.public.20100101-20160101.pkl")
    df = pd.read_pickle(data_path)
    compute_clearsky_rmse(df)


if __name__ == "__main__":
    main()
