import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from models.CNN2D import CNN2D
from src.data_pipeline import hdf5_dataloader_list_of_days
from src.schema import Catalog

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

epochs = 25
data_path = Path(Path(__file__).parent.parent, "data")
df = pd.read_pickle(Path(data_path, "catalog.helios.public.20100101-20160101.pkl"))

# Quick fix to avoid nan, should be handled better
# TODO make sure training and validation examples are valid
hdf5_paths = df[Catalog.hdf5_8bit_path]
df = df.fillna(0)
df[Catalog.hdf5_8bit_path] = hdf5_paths

target_time_offsets = [datetime.timedelta(hours=0), datetime.timedelta(hours=1), datetime.timedelta(hours=3),
                       datetime.timedelta(hours=6)]

start_time = "2011-01-01T08:00:00"
start_datetime = datetime.datetime.fromisoformat(start_time)
num_days = 24
train_target_datetimes = [start_datetime + datetime.timedelta(days=x) for x in range(num_days)]
train_data = hdf5_dataloader_list_of_days(df, train_target_datetimes,
                                          target_time_offsets, data_directory=Path(data_path, "hdf5v7_8bit"),
                                          batch_size=16, test_time=False)

start_val_time = "2011-01-25T08:00:00"
val_start_datetime = datetime.datetime.fromisoformat(start_val_time)
val_num_days = 7
val_target_datetimes = [val_start_datetime + datetime.timedelta(days=x) for x in range(val_num_days)]

val_data = hdf5_dataloader_list_of_days(df, val_target_datetimes,
                                        target_time_offsets, data_directory=Path(data_path, "hdf5v7_8bit"),
                                        batch_size=16, test_time=False)

model = CNN2D()
metadata_len = 4 + len(target_time_offsets)
model.build([(None, 32, 32,
              5), (None, metadata_len)])

model.compile(loss='mse',
              optimizer='adam')

history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data)

model.save_weights("saved_models/cnn2d")

plt.plot(range(1, epochs + 1), history.history['loss'], label='train_loss')
plt.plot(range(1, epochs + 1), history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
