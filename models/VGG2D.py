"""
Basic model example using a 2D CNN with 32x32x3 images as inputs and metadata
Used only for demonstration purposes and not to be used on real datas
"""
import tensorflow as tf


class VGG2D(tf.keras.Model):
    """
    Model based on paper
    A deep learning approach to solar-irradiance forecasting in sky
    https://arxiv.org/abs/1901.04881
    ** Not Finished **
    """
    def __init__(self):
        """
        Define model layers
        """
        super(VGG2D, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(64, (7, 7),
                                             activation='relu',
                                             padding='same',
                                             name='block1_conv1',
                                             dilation_rate=(4, 4))

        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3),
                                             activation='relu',
                                             padding='same',
                                             name='block1_conv2')

        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        # Block 2
        self.conv_3 = tf.keras.layers.Conv2D(128, (3, 3),
                                             activation='relu',
                                             padding='same',
                                             name='block2_conv1')
        self.conv_4 = tf.keras.layers.Conv2D(128, (3, 3),
                                             activation='relu',
                                             padding='same',
                                             name='block2_conv2')

        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        self.conv_5 = tf.keras.layers.Conv2D(256, (3, 3),
                                             activation='relu',
                                             padding='same',
                                             name='block3_conv1')
        self.conv_6 = tf.keras.layers.Conv2D(256, (3, 3),
                                             activation='relu',
                                             padding='same',
                                             name='block3_conv2')
        self.conv_7 = tf.keras.layers.Conv2D(256, (3, 3),
                                             activation='relu',
                                             padding='same',
                                             name='block3_conv3')

        self.max_pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        # Top
        self.flatten = tf.keras.layers.Flatten()
        self.FC1 = tf.keras.layers.Dense(512, activation='relu')
        self.FC2 = tf.keras.layers.Dense(256, activation='relu')
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
        # Block 1
        x = self.conv_1(img)
        x = self.conv_2(x)
        x = self.max_pool1(x)

        # Block 2
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.max_pool2(x)

        # Block 3
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.max_pool3(x)

        # Top
        x = self.flatten(x)
        # concatenate encoded image and metadata
        x = tf.keras.layers.concatenate([x, past_metadata, future_metadata], 1)
        x = self.FC1(x)
        x = self.FC2(x)
        # Create 4 outputs for t0, t0+1, t0+3 and t0+6
        t0 = self.t0(x)
        return t0
