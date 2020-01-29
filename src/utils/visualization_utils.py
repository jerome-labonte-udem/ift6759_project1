import datetime
import json
import math
import os
import typing
import warnings

import cv2 as cv
import h5py
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from src.hdf5 import HDF5File


def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""
    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0
    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


def get_label_html_color_code(idx):
    """Returns the PASCAL VOC HTML color code for a given label index."""
    color_array = get_label_color_mapping(idx)
    return f"#{color_array[0]:02X}{color_array[1]:02X}{color_array[2]:02X}"


def fig2array(fig):
    """Transforms a pyplot figure into a numpy-compatible BGR array.
    The reason why we flip the channel order (RGB->BGR) is for OpenCV compatibility. Feel free to
    edit this function if you wish to use it with another display library.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf[..., ::-1]


def viz_hdf5_imagery(
        hdf5_path: str,
        channels: typing.List[str],
        dataframe_path: typing.Optional[str] = None,
        stations: typing.Optional[typing.Dict[str, typing.Tuple]] = None,
        copy_last_if_missing: bool = True,
) -> None:
    """Displays a looping visualization of the imagery channels saved in an HDF5 file.
    This visualization requires OpenCV3+ ('cv2'), and will loop while refreshing a local window until the program
    is killed, or 'q' is pressed. The visualization can also be paused by pressing the space bar.
    """
    assert os.path.isfile(hdf5_path), f"invalid hdf5 path: {hdf5_path}"
    assert channels, "list of channels must not be empty"
    with h5py.File(hdf5_path, "r") as f_h5_data:
        h5 = HDF5File(f_h5_data)
        lut_timestamps = h5.lut_time_stamps()
        # will only display GHI values if dataframe is available
        stations_data = {}
        if stations:
            df = pd.read_pickle(dataframe_path) if dataframe_path else None
            # assume lats/lons stay identical throughout all frames; just pick the first available arrays
            idx, lats, lons = 0, None, None
            while (lats is None or lons is None) and idx < h5.archive_lut_size:
                lats, lons = h5.fetch_lat(idx), h5.fetch_long(idx)
            assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"
            for reg, coords in tqdm.tqdm(stations.items(), desc="preparing stations data"):
                station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
                station_data = {"coords": station_coords}
                if dataframe_path:
                    station_data["ghi"] = [df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]
                    station_data["csky"] = [df.at[pd.Timestamp(t), reg + "_CLEARSKY_GHI"] for t in lut_timestamps]
                stations_data[reg] = station_data
        print(stations_data)
        raw_data = np.zeros((h5.archive_lut_size, len(channels), 650, 1500, 3), dtype=np.uint8)
        for channel_idx, channel_name in tqdm.tqdm(enumerate(channels), desc="preparing img data", total=len(channels)):
            assert channel_name in h5.file, f"missing channel: {channels}"
            norm_min = h5.orig_min(channel_name)
            norm_max = h5.orig_max(channel_name)
            channel_data = [h5.fetch_sample(channel_name, idx) for idx in range(h5.archive_lut_size)]
            assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
                "one of the saved channels had an expected dimension"
            last_valid_array_idx = None
            for array_idx, array in enumerate(channel_data):
                if array is None:
                    if copy_last_if_missing and last_valid_array_idx is not None:
                        raw_data[array_idx, channel_idx, :, :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                    continue
                array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                for station_idx, (station_name, station) in enumerate(stations_data.items()):
                    station_color = get_label_color_mapping(station_idx + 1).tolist()[::-1]
                    array = cv.circle(array, station["coords"][::-1], radius=9, color=station_color, thickness=-1)
                raw_data[array_idx, channel_idx, :, :] = cv.flip(array, 0)
                last_valid_array_idx = array_idx
    plot_data = None
    if stations and dataframe_path:
        plot_data = preplot_live_ghi_curves(
            stations=stations, stations_data=stations_data,
            window_start=h5.start_time,
            window_end=h5.start_time + datetime.timedelta(hours=24),
            sample_step=datetime.timedelta(minutes=15),
            plot_title=h5.start_time.strftime("GHI @ %Y-%m-%d"),
        )
        assert plot_data.shape[0] == h5.archive_lut_size
    display_data = []
    for array_idx in tqdm.tqdm(range(h5.archive_lut_size), desc="reshaping for final display"):
        display = cv.vconcat([raw_data[array_idx, ch_idx, ...] for ch_idx in range(len(channels))])
        while any([s > 1200 for s in display.shape]):
            display = cv.resize(display, (-1, -1), fx=0.75, fy=0.75)
        if plot_data is not None:
            plot = plot_data[array_idx]
            plot_scale = display.shape[0] / plot.shape[0]
            plot = cv.resize(plot, (-1, -1), fx=plot_scale, fy=plot_scale)
            display = cv.hconcat([display, plot])
        display_data.append(display)
    display = np.stack(display_data)
    array_idx, window_name, paused = 0, hdf5_path.split("/")[-1], False
    while True:
        cv.imshow(window_name, display[array_idx])
        ret = cv.waitKey(30 if not paused else 300)
        if ret == ord('q') or ret == 27:  # q or ESC
            break
        elif ret == ord(' '):
            paused = ~paused
        if not paused or ret == ord('c'):
            array_idx = (array_idx + 1) % h5.archive_lut_size


def preplot_live_ghi_curves(
        stations: typing.Dict[str, typing.Tuple],
        stations_data: typing.Dict[str, typing.Any],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
        plot_title: typing.Optional[typing.AnyStr] = None,
) -> np.ndarray:
    """Pre-plots a set of GHI curves with update bars and returns the raw pixel arrays.

    This function is used in ``viz_hdf5_imagery`` to prepare GHI plots when stations & dataframe information
    is available.
    """
    plot_count = (window_end - window_start) // sample_step
    fig_size, fig_dpi, plot_row_count = (8, 6), 160, int(math.ceil(len(stations) / 2))
    plot_data = np.zeros((plot_count, fig_size[0] * fig_dpi, fig_size[1] * fig_dpi, 3), dtype=np.uint8)
    fig = plt.figure(num="ghi", figsize=fig_size[::-1], dpi=fig_dpi, facecolor="w", edgecolor="k")
    ax = fig.subplots(nrows=plot_row_count, ncols=2, sharex="all", sharey="all")
    art_handles, art_labels = [], []
    for station_idx, station_name in enumerate(stations):
        plot_row_idx, plot_col_idx = station_idx // 2, station_idx % 2
        ax[plot_row_idx, plot_col_idx] = plot_ghi_curves(
            clearsky_ghi=np.asarray(stations_data[station_name]["csky"]),
            station_ghi=np.asarray(stations_data[station_name]["ghi"]),
            pred_ghi=None,
            window_start=window_start,
            window_end=window_end - sample_step,
            sample_step=sample_step,
            horiz_offset=datetime.timedelta(hours=0),
            ax=ax[plot_row_idx, plot_col_idx],
            station_name=station_name,
            station_color=get_label_html_color_code(station_idx + 1),
            current_time=window_start
        )
        for handle, lbl in zip(*ax[plot_row_idx, plot_col_idx].get_legend_handles_labels()):
            # skipping over the duplicate labels messes up the legend, we must live with the warning
            art_labels.append("_" + lbl if lbl in art_labels or lbl == "current" else lbl)
            art_handles.append(handle)
    fig.autofmt_xdate()
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.legend(art_handles, labels=art_labels, loc="lower center", ncol=2)
    fig.canvas.draw()  # cache renderer with default call first
    subaxbgs = [fig.canvas.copy_from_bbox(subax.bbox) for subax in ax.flatten()]
    for idx in tqdm.tqdm(range(plot_count), desc="preparing ghi plots"):
        for subax, subaxbg in zip(ax.flatten(), subaxbgs):
            fig.canvas.restore_region(subaxbg)
            for handle, lbl in zip(*subax.get_legend_handles_labels()):
                if lbl == "current":
                    curr_time = matplotlib.dates.date2num(window_start + idx * sample_step)
                    handle.set_data([curr_time, curr_time], [0, 1])
                    subax.draw_artist(handle)
            fig.canvas.blit(subax.bbox)
        plot_data[idx, ...] = np.reshape(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8),
                                         (*(fig.canvas.get_width_height()[::-1]), 3))[..., ::-1]
    return plot_data


def plot_ghi_curves(
        clearsky_ghi: np.ndarray,
        station_ghi: np.ndarray,
        pred_ghi: typing.Optional[np.ndarray],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
        horiz_offset: datetime.timedelta,
        ax: plt.Axes,
        station_name: typing.Optional[typing.AnyStr] = None,
        station_color: typing.Optional[typing.AnyStr] = None,
        current_time: typing.Optional[datetime.datetime] = None,
) -> plt.Axes:
    """Plots a set of GHI curves and returns the associated matplotlib axes object.

    This function is used in ``draw_daily_ghi`` and ``preplot_live_ghi_curves`` to create simple
    graphs of GHI curves (clearsky, measured, predicted).
    """
    assert clearsky_ghi.ndim == 1 and station_ghi.ndim == 1 and clearsky_ghi.size == station_ghi.size
    assert pred_ghi is None or (pred_ghi.ndim == 1 and clearsky_ghi.size == pred_ghi.size)
    hour_tick_locator = matplotlib.dates.HourLocator(interval=4)
    minute_tick_locator = matplotlib.dates.HourLocator(interval=1)
    datetime_fmt = matplotlib.dates.DateFormatter("%H:%M")
    datetime_range = pd.date_range(window_start, window_end, freq=sample_step)
    xrange_real = matplotlib.dates.date2num([d.to_pydatetime() for d in datetime_range])
    if current_time is not None:
        ax.axvline(x=matplotlib.dates.date2num(current_time), color="r", label="current")
    station_name = f"measured ({station_name})" if station_name else "measured"
    ax.plot(xrange_real, clearsky_ghi, ":", label="clearsky")
    if station_color is not None:
        ax.plot(xrange_real, station_ghi, linestyle="solid", color=station_color, label=station_name)
    else:
        ax.plot(xrange_real, station_ghi, linestyle="solid", label=station_name)
    datetime_range = pd.date_range(window_start + horiz_offset, window_end + horiz_offset, freq=sample_step)
    xrange_offset = matplotlib.dates.date2num([d.to_pydatetime() for d in datetime_range])
    if pred_ghi is not None:
        ax.plot(xrange_offset, pred_ghi, ".-", label="predicted")
    ax.xaxis.set_major_locator(hour_tick_locator)
    ax.xaxis.set_major_formatter(datetime_fmt)
    ax.xaxis.set_minor_locator(minute_tick_locator)
    hour_offset = datetime.timedelta(hours=1) // sample_step
    ax.set_xlim(xrange_real[hour_offset - 1], xrange_real[-hour_offset + 1])
    ax.format_xdata = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    ax.grid(True)
    return ax


def draw_daily_ghi(
        clearsky_ghi: np.ndarray,
        station_ghi: np.ndarray,
        pred_ghi: np.ndarray,
        stations: typing.Iterable[typing.AnyStr],
        horiz_deltas: typing.List[datetime.timedelta],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
):
    """Draws a set of 2D GHI curve plots and returns the associated matplotlib fig/axes objects.

    This function is used in ``viz_predictions`` to prepare the full-horizon, multi-station graphs of
    GHI values over numerous days.
    """
    assert clearsky_ghi.ndim == 2 and station_ghi.ndim == 2 and clearsky_ghi.shape == station_ghi.shape
    station_count = len(list(stations))
    sample_count = station_ghi.shape[1]
    assert clearsky_ghi.shape[0] == station_count and station_ghi.shape[0] == station_count
    assert pred_ghi.ndim == 3 and pred_ghi.shape[0] == station_count and pred_ghi.shape[2] == sample_count
    assert len(list(horiz_deltas)) == pred_ghi.shape[1]
    pred_horiz = pred_ghi.shape[1]
    fig = plt.figure(num="ghi", figsize=(18, 10), dpi=80, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.subplots(nrows=pred_horiz, ncols=station_count, sharex="all", sharey="all")
    handles, labels = None, None
    for horiz_idx in range(pred_horiz):
        for station_idx, station_name in enumerate(stations):
            ax[horiz_idx, station_idx] = plot_ghi_curves(
                clearsky_ghi=clearsky_ghi[station_idx],
                station_ghi=station_ghi[station_idx],
                pred_ghi=pred_ghi[station_idx, horiz_idx],
                window_start=window_start,
                window_end=window_end,
                sample_step=sample_step,
                horiz_offset=horiz_deltas[horiz_idx],
                ax=ax[horiz_idx, station_idx],
            )
            handles, labels = ax[horiz_idx, station_idx].get_legend_handles_labels()
    for station_idx, station_name in enumerate(stations):
        ax[0, station_idx].set_title(station_name)
    for horiz_idx, horiz_delta in zip(range(pred_horiz), horiz_deltas):
        ax[horiz_idx, 0].set_ylabel(f"GHI @ T+{horiz_delta}")
    window_center = window_start + (window_end - window_start) / 2
    fig.autofmt_xdate()
    fig.suptitle(window_center.strftime("%Y-%m-%d"), fontsize=14)
    fig.legend(handles, labels, loc="lower center")
    return fig2array(fig)


def viz_predictions(
        predictions_path: typing.AnyStr,
        dataframe_path: typing.AnyStr,
        test_config_path: typing.AnyStr,
):
    """Displays a looping visualization of the GHI predictions saved by the evaluation script.

    This visualization requires OpenCV3+ ('cv2'), and will loop while refreshing a local window until the program
    is killed, or 'q' is pressed. The arrow keys allow the user to change which day is being shown.
    """
    assert os.path.isfile(test_config_path) and test_config_path.endswith(".json"), "invalid test config"
    with open(test_config_path, "r") as fd:
        test_config = json.load(fd)
    stations = test_config["stations"]
    target_datetimes = test_config["target_datetimes"]
    start_bound = datetime.datetime.fromisoformat(test_config["start_bound"])
    end_bound = datetime.datetime.fromisoformat(test_config["end_bound"])
    horiz_deltas = [pd.Timedelta(d).to_pytimedelta() for d in test_config["target_time_offsets"]]
    assert os.path.isfile(predictions_path), f"invalid preds file path: {predictions_path}"
    with open(predictions_path, "r") as fd:
        predictions = fd.readlines()
    assert len(predictions) == len(target_datetimes) * len(stations), \
        "predicted ghi sequence count mistmatch wrt target datetimes x station count"
    assert len(predictions) % len(stations) == 0
    predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    predictions = predictions.reshape((len(stations), len(target_datetimes), -1))
    pred_horiz = predictions.shape[-1]
    target_datetimes = pd.DatetimeIndex([datetime.datetime.fromisoformat(t) for t in target_datetimes])
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)
    dataframe = dataframe[dataframe.index >= start_bound]
    dataframe = dataframe[dataframe.index < end_bound]
    assert dataframe.index.get_loc(start_bound) == 0, "invalid start bound (should land at first index)"
    assert len(dataframe.index.intersection(target_datetimes)) == len(target_datetimes), \
        "bad dataframe target datetimes overlap, index values are missing"
    # we will display 24-hour slices with some overlap (configured via hard-coded param below)
    time_window, time_overlap, time_sample = \
        datetime.timedelta(hours=24), datetime.timedelta(hours=3), datetime.timedelta(minutes=15)
    assert len(dataframe.asfreq("15min").index) == len(dataframe.index), \
        "invalid dataframe index padding (should have an entry every 15 mins)"
    sample_count = ((time_window + 2 * time_overlap) // time_sample) + 1
    day_count = int(math.ceil((end_bound - start_bound) / time_window))
    clearsky_ghi_data = np.full((day_count, len(stations), sample_count), fill_value=float("nan"), dtype=np.float32)
    station_ghi_data = np.full((day_count, len(stations), sample_count), fill_value=float("nan"), dtype=np.float32)
    pred_ghi_data = np.full((day_count, len(stations), pred_horiz, sample_count),
                            fill_value=float("nan"), dtype=np.float32)
    days_range = pd.date_range(start_bound, end_bound, freq=time_window, closed="left")
    for day_idx, day_start in enumerate(tqdm.tqdm(days_range, desc="preparing daytime GHI intervals")):
        window_start, window_end = day_start - time_overlap, day_start + time_window + time_overlap
        sample_start = (window_start - start_bound) // time_sample
        sample_end = (window_end - start_bound) // time_sample
        for sample_iter_idx, sample_idx in enumerate(range(sample_start, sample_end + 1)):
            if sample_idx < 0 or sample_idx >= len(dataframe.index):
                continue
            sample_row = dataframe.iloc[sample_idx]
            sample_time = window_start + sample_iter_idx * time_sample
            target_iter_idx = target_datetimes.get_loc(sample_time) if sample_time in target_datetimes else None
            for station_idx, station_name in enumerate(stations):
                clearsky_ghi_data[day_idx, station_idx, sample_iter_idx] = sample_row[station_name + "_CLEARSKY_GHI"]
                station_ghi_data[day_idx, station_idx, sample_iter_idx] = sample_row[station_name + "_GHI"]
                if target_iter_idx is not None:
                    pred_ghi_data[day_idx, station_idx, :, sample_iter_idx] = predictions[station_idx, target_iter_idx]
    displays = []
    for day_idx, day_start in enumerate(tqdm.tqdm(days_range, desc="preparing plots")):
        displays.append(draw_daily_ghi(
            clearsky_ghi=clearsky_ghi_data[day_idx],
            station_ghi=station_ghi_data[day_idx],
            pred_ghi=pred_ghi_data[day_idx],
            stations=stations,
            horiz_deltas=horiz_deltas,
            window_start=(day_start - time_overlap),
            window_end=(day_start + time_window + time_overlap),
            sample_step=time_sample,
        ))
    display = np.stack(displays)
    day_idx = 0
    while True:
        cv.imshow("ghi", display[day_idx])
        ret = cv.waitKey(100)
        if ret == ord('q') or ret == 27:  # q or ESC
            break
        elif ret == 81 or ret == 84:  # UNIX: left or down arrow
            day_idx = max(day_idx - 1, 0)
        elif ret == 82 or ret == 83:  # UNIX: right or up arrow
            day_idx = min(day_idx + 1, len(displays) - 1)
