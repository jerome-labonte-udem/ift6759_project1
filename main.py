from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import datasets

from models.CNN2D import CNN2D
from src.data_utils import get_labels, get_metadata

train_size = 5000  # used only to limit the size for debugging purpose
val_size = 1000

# Temporarily use cifar10 images before real images dataloader is available
(train_images, _), (val_images, _) = datasets.cifar10.load_data()

# train_size examples should be chosen at random here
train_images = train_images[:train_size]
# val_size examples should be chosen at random here
val_images = val_images[:val_size]

# Normalize pixel values to be between 0 and 1
train_images, val_images = train_images[:train_size] / 255.0,\
                           val_images[:val_size] / 255.0

data_path = Path("data")  # redefine data path here if needed
df = pd.read_pickle(Path(data_path,
                         "catalog.helios.public.20100101-20160101.pkl"))

train_labels = get_labels(df, "BND", "2010-05-01", "2010-07-01")[:train_size]

# Run clearsky baseline before handling nan
train_metadata = get_metadata(df, "BND", "2010-05-01",
                              "2010-07-01")[:train_size]
train_clearsky = train_metadata[:, 0]
val_metadata = get_metadata(df, "BND", "2010-07-01",
                            "2010-09-01")[:val_size]
val_clearsky = val_metadata[:, 0]
# Get average error on t0 only as it should be the same for future predictions
train_clearsky_mse = np.nanmean((train_labels[:, 0] - train_clearsky)**2)

df = df.fillna(0)  # Replace nan by 0 for now, should be handled correctly !!!

# get labels and metadata
"""
metadata contain : CLEARSKY_GHI, DAYTIME (bool), day of year, hour, minute
the method get_metada should be modified to change the type of data if needed
"""
train_metadata = get_metadata(df, "BND", "2010-05-01",
                              "2010-07-01")[:train_size]
val_metadata = get_metadata(df, "BND", "2010-07-01", "2010-09-01")[:val_size]

train_labels = get_labels(df, "BND", "2010-05-01", "2010-07-01")[:train_size]
val_labels = bnd_labels = get_labels(df, "BND", "2010-07-01",
                                     "2010-09-01")[:val_size]

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
                    epochs=5,
                    validation_data=([val_images, val_metadata],
                                     [val_labels[:, 0],
                                      val_labels[:, 1],
                                      val_labels[:, 2],
                                      val_labels[:, 3]]))


plt.plot(history.history['output_1_mse'], label='t0_mse')
plt.plot(history.history['output_2_mse'], label='t1_mse')
plt.plot(history.history['output_3_mse'], label='t2_mse')
plt.plot(history.history['output_4_mse'], label='t3_mse')
plt.plot([train_clearsky_mse] * len(history.history['output_1_mse']),
         label="clearsky_mse")
plt.xlabel('Epoch')
plt.ylabel('train MSE')
plt.legend()
plt.show()
