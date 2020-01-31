import h5py
from src.hdf5 import HDF5File
import datetime
from src.schema import Catalog
from src.data_pipeline import hdf5_dataloader_list_of_days
from pathlib import Path
import pandas as pd
import numpy as np

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

epochs = 1
data_path = Path(Path(__file__).parent.parent, "data")  # redefine data path here if needed
df = pd.read_pickle(Path(data_path,
                         "catalog.helios.public.20100101-20160101.pkl"))

print(df.head())
hdf5_paths = df[Catalog.hdf5_8bit_path]
df = df.fillna(0)
df[Catalog.hdf5_8bit_path] = hdf5_paths
print(df.head())
train_target_datetimes = pd.to_datetime(df.index[70368:70368 + 10])
val_target_datetimes = pd.to_datetime(df.index[70368 + 10:70368 + 30])
# Only test on using t0 for label
target_time_offsets = [datetime.timedelta(hours=0), datetime.timedelta(hours=0), datetime.timedelta(hours=0),
                       datetime.timedelta(hours=0)]
train_data = hdf5_dataloader_list_of_days(df, train_target_datetimes,
                                          target_time_offsets, data_directory=Path(data_path, "hdf5v7_8bit"),
                                          batch_size=32, test_time=False)
val_data = hdf5_dataloader_list_of_days(df, val_target_datetimes,
                                        target_time_offsets, data_directory=Path(data_path, "hdf5v7_8bit"),
                                        batch_size=96, test_time=False)

from models.CNN2D import CNN2D

model = CNN2D()
metadata_len = 4 + len(target_time_offsets)
model.build([(None, 32, 32,
              5), (None, metadata_len)])

model.compile(loss='mse',
              optimizer='adam')

history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data)

import matplotlib.pyplot as plt

model.save("saved_models/cnn2d")

plt.plot(range(1, epochs + 1), history.history['loss'], label='train_loss')
plt.plot(range(1, epochs + 1), history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
