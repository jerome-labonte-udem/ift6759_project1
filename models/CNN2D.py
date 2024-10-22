"""
Basic model example using a 2D CNN with images as inputs and metadata
"""
import tensorflow as tf


class CNN2D(tf.keras.Model):
    """
    Simple 2D CNN
    """
    def __init__(self):
        """
        Define model layers
        """
        super(CNN2D, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool_1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.FC1 = tf.keras.layers.Dense(128, activation='relu')
        self.FC2 = tf.keras.layers.Dense(64, activation='relu')

        self.t0 = tf.keras.layers.Dense(4, name='t0')

    def call(self, inputs):
        """
        Perform forward on inputs and returns predicted GHI at the
        desired times
        :param inputs: input images and metadata
        :return: a list of four floats for GHI at each desired time
        """
        img, past_metadata, future_metadata = inputs  # split images and metadatas
        # Remove timesteps dimensions from sequences of size 1
        patch_size = img.shape[-2]
        n_channels = img.shape[-1]
        img = tf.reshape(img, (-1, patch_size, patch_size, n_channels))
        past_metadata = tf.reshape(past_metadata, (-1, past_metadata.shape[-1]))
        x = self.conv_1(img)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        # concatenate encoded image and metadata
        x = tf.keras.layers.concatenate([x, past_metadata, future_metadata], 1)
        x = self.FC1(x)
        x = self.FC2(x)
        # Create 4 outputs for t0, t0+1, t0+3 and t0+6
        t0 = self.t0(x)
        return t0
