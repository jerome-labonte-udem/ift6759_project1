"""
Basic model example using a 2D CNN with 32x32x3 images as inputs and metadata
Used only for demonstration purposes and not to be used on real datas
"""
import tensorflow as tf


class CNN2D(tf.keras.Model):
    """
    Toy model to test multiple inputs and outputs
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
        img, metadata = inputs  # split images and metadata
        x = self.conv_1(img)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        # concatenate encoded image and metadata
        x = tf.keras.layers.concatenate([x, metadata], 1)
        x = self.FC1(x)
        x = self.FC2(x)
        # Create 4 outputs for t0, t0+1, t0+3 and t0+6
        t0 = self.t0(x)

        return t0
