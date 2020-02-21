import tensorflow as tf


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, feat_maps_in: int, feat_maps_out: int):
        super(ResBlock, self).__init__()
        stride = 1 if feat_maps_in == feat_maps_out else 2
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv2D(feat_maps_out, (3, 3),
                                            strides=(stride, stride), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(feat_maps_out, (3, 3), padding='same')
        self.skip = tf.keras.layers.Conv2D(feat_maps_out, (1, 1),
                                           strides=(stride, stride), padding='valid')
        self.merge = tf.keras.layers.Add()

    def call(self, inputs):
        # skip connection
        identity = self.skip(inputs)
        # conv connection
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # merge + activation
        x = self.merge([x, identity])
        x = self.act2(x)
        return x


class ResNet3D(tf.keras.Model):
    def __init__(self):
        """
        Define model layers
        """
        super(ResNet3D, self).__init__()
        self.conv_1 = tf.keras.layers.Conv3D(
            32, (5, 7, 7), input_shape=(None, 5, 64, 64, 5), activation="relu"
        )
        self.res_block1 = ResBlock(32, 32)
        self.res_block2 = ResBlock(32, 64)
        self.res_block3 = ResBlock(64, 64)
        self.res_block4 = ResBlock(64, 128)
        self.res_block5 = ResBlock(128, 128)

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # for pastmetadata
        self.rnn = tf.keras.layers.SimpleRNN(8, dropout=0.1, recurrent_dropout=0.1)

        # self.FC1 = tf.keras.layers.Dense(64, activation='relu')
        self.t0 = tf.keras.layers.Dense(4, name='t0')

    def call(self, inputs):
        """
        Perform forward on inputs and returns predicted GHI at the
        desired times
        :param inputs: input images and metadata
        :return: a list of four floats for GHI at each desired time
        """
        img, past_metadata, future_metadata = inputs  # split images and metadatas
        x = self.conv_1(img)
        x = tf.squeeze(x, axis=1)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.avg_pool(x)

        # rnn for past metadata
        pmd = self.rnn(past_metadata)

        x = tf.keras.layers.concatenate([x, pmd, future_metadata], 1)
        # x = self.FC1(x)
        # Create 4 outputs for t0, t0+1, t0+3 and t0+6
        t0 = self.t0(x)
        return t0
