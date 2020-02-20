import tensorflow as tf


class CNN3D(tf.keras.Model):
    """
    Toy model to test multiple inputs and outputs
    """

    def __init__(self):
        """
        Define model layers
        """
        super(CNN3D, self).__init__()
        self.conv_3d = tf.keras.layers.Conv3D(64, (5, 7, 7), input_shape=(None, 5, 64, 64, 5),
                                              strides=(1, 1, 1), activation="relu")

        # squeeze here to go to 2d
        self.conv_2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')

        self.conv_4 = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # for pastmetadata
        self.rnn = tf.keras.layers.SimpleRNN(10, dropout=0.1, recurrent_dropout=0.1)

        # self.flatten = tf.keras.layers.Flatten()
        self.FC1 = tf.keras.layers.Dense(64, activation='relu')
        # self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.t0 = tf.keras.layers.Dense(4, name='t0')

    def call(self, inputs):
        """
        Perform forward on inputs and returns predicted GHI at the
        desired times
        :param inputs: input images and metadata
        :return: a list of four floats for GHI at each desired time
        """
        img, past_metadata, future_metadata = inputs  # split images and metadatas

        x = self.conv_3d(img)
        x = tf.squeeze(x, axis=1)

        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        # x = self.flatten(x)
        x = self.avg_pool(x)
        # rnn for past metadata
        pmd = self.rnn(past_metadata)
        # concatenate encoded image and metadata

        x = tf.keras.layers.concatenate([x, pmd, future_metadata], 1)
        x = self.FC1(x)
        # Create 4 outputs for t0, t0+1, t0+3 and t0+6
        t0 = self.t0(x)
        return t0
