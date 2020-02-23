import tensorflow as tf


class ResBlock(tf.keras.Model):

    def __init__(self, feat_maps_in: int, feat_maps_out: int):
        super(ResBlock, self).__init__()
        stride = 1 if feat_maps_in == feat_maps_out else 2
        self.conv1 = tf.keras.layers.Conv2D(feat_maps_out, (3, 3),
                                            strides=(stride, stride), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(feat_maps_out, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if feat_maps_out != feat_maps_in:
            self.downsample = tf.keras.layers.Conv2D(feat_maps_out, (1, 1),
                                                     strides=(stride, stride), padding='valid')
        else:
            self.downsample = None

    def call(self, input_tensor):
        # first layer
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        # second layer
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            input_tensor = self.downsample(input_tensor)

        x += input_tensor
        return tf.nn.relu(x)


class ResNet3D(tf.keras.Model):
    def __init__(self):
        """
        Define model layers
        """
        super(ResNet3D, self).__init__()
        self.conv_1 = tf.keras.layers.Conv3D(
            32, (3, 7, 7), input_shape=(None, 3, 64, 64, 5), strides=(1, 2, 2)
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.res_block1a = ResBlock(32, 32)
        self.res_block1b = ResBlock(32, 32)
        self.res_block1c = ResBlock(32, 32)

        self.res_block2a = ResBlock(32, 64)
        self.res_block2b = ResBlock(64, 64)
        self.res_block2c = ResBlock(64, 64)

        self.res_block3a = ResBlock(64, 128)
        self.res_block3b = ResBlock(128, 128)
        self.res_block3c = ResBlock(128, 128)

        # self.flatten = tf.keras.layers.Flatten()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # for pastmetadata
        self.rnn = tf.keras.layers.SimpleRNN(8, dropout=0.1, recurrent_dropout=0.1)

        self.FC1 = tf.keras.layers.Dense(128, activation='relu')
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
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.res_block1a(x)
        x = self.res_block1b(x)
        x = self.res_block1c(x)
        print(f"res_block1c.shape = {x.shape} ")
        x = self.res_block2a(x)
        x = self.res_block2b(x)
        x = self.res_block2c(x)
        print(f"res_block2c.shape = {x.shape} ")
        x = self.res_block3a(x)
        print(f"res_block3a.shape = {x.shape} ")
        x = self.res_block3b(x)
        print(f"res_block3b.shape = {x.shape} ")
        x = self.res_block3c(x)
        print(f"res_block3c.shape = {x.shape} ")

        x = self.avg_pool(x)
        print(f"avg_pool.shape = {x.shape} ")
        # x = self.flatten(x)
        # rnn for past metadata
        pmd = self.rnn(past_metadata)

        x = tf.keras.layers.concatenate([x, pmd, future_metadata], 1)
        print(f"concatenate.shape = {x.shape} ")
        x = self.FC1(x)
        # Create 4 outputs for t0, t0+1, t0+3 and t0+6
        t0 = self.t0(x)
        return t0
