from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.CNN2D import CNN2D
from src.data_utils import get_labels, get_metadata

data_path = Path("data")  # redefine data path here if needed
df = pd.read_pickle(Path(data_path,
                         "catalog.helios.public.20100101-20160101.pkl"))

train_labels = get_labels(df, "BND", "2011-01-01", "2011-12-31")
val_labels = get_labels(df, "BND", "2012-01-01", "2012-12-31")

# Run clearsky baseline before handling nan
train_metadata = get_metadata(df, "BND", "2011-01-01",
                              "2011-12-31")
train_clearsky = train_metadata[:, 0]
val_metadata = get_metadata(df, "BND", "2012-01-01",
                            "2012-12-31")
val_clearsky = val_metadata[:, 0]
# Get average error on t0 only as it should be the same for future predictions
train_clearsky_mse = np.nanmean((train_labels[:, 0] - train_clearsky)**2)
val_clearsky_mse = np.nanmean((val_labels[:, 0] - val_clearsky)**2)
print("train clearksy mse: ", train_clearsky_mse)
print("val clearsky mse: ", val_clearsky_mse)


df = df.fillna(0)  # Replace nan by 0 for now, should be handled correctly !!!
# images are zeros for now both should be real images
train_size = train_metadata.shape[0]
val_size = val_metadata.shape[0]
# train_size = 5000  # used only to limit the size for debugging purpose
# val_size = 1000
train_images = np.zeros((train_size, 32, 32, 5))
val_images = np.zeros((val_size, 32, 32, 5))

# get labels and metadata
"""
metadata contain : CLEARSKY_GHI, CLEARSKY_GHI at t+1 hour,
CLEARSKY_GHI at t+3 hours, CLEARSKY_GHI at t+6 hours,
DAYTIME (bool), day of year, hour, minute
the method get_metada should be modified to change the type of data if needed
"""
train_metadata = get_metadata(df, "BND", "2011-01-01",
                              "2011-12-31")
val_metadata = get_metadata(df, "BND", "2012-01-01", "2012-12-31")

train_labels = get_labels(df, "BND", "2011-01-01", "2011-12-31")
val_labels = get_labels(df, "BND", "2012-01-01", "2012-12-31")

# create model
model = CNN2D()
model.compile(loss=['mse', 'mse', 'mse', 'mse'],
              loss_weights=[0.25, 0.25, 0.25, 0.25],
              optimizer='adam',
              metrics=['mse', 'mse', 'mse', 'mse'])

# Build model with inputs shapes to initialize weights
model.build([(None, train_images.shape[1], train_images.shape[2],
              train_images.shape[3]),
             (None, train_metadata.shape[1])])
# train model
print(model.summary())

# Fit model with firt 5000 examples
history = model.fit([train_images, train_metadata],
                    [train_labels[:, 0],
                     train_labels[:, 1],
                     train_labels[:, 2],
                     train_labels[:, 3]],
                    epochs=10,
                    validation_data=([val_images, val_metadata],
                                     [val_labels[:, 0],
                                      val_labels[:, 1],
                                      val_labels[:, 2],
                                      val_labels[:, 3]]))


plt.plot(range(1, 11), history.history['val_loss'], label='val_loss')
plt.plot(range(1, 11), history.history['loss'], label='train_loss')
plt.plot(range(1, 11),
         [val_clearsky_mse] * len(history.history['val_output_1_mse']),
         label="val_clearsky_mse")
plt.plot(range(1, 11),
         [train_clearsky_mse] * len(history.history['val_output_1_mse']),
         label="train_clearsky_mse")
plt.xlabel('Epoch')
plt.ylabel('val MSE')
plt.legend()
plt.show()
