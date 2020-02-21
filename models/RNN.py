"""
RNN model with a Custom cell that takes a sequence of images and metadas
"""

import tensorflow as tf


class CustomCell(tf.keras.layers.Layer):
    """
    RNN cell that applies convolution to image and concatenate it to metadatas
    """
    def __init__(self,
                 units: int,
                 activation: str = 'tanh',
                 kernel_initializer: str = 'glorot_uniform',
                 recurrent_initializer: str = 'identity',
                 ) -> None:
        """
        Init method
        :param units: size of hidden state
        :param activation: activation method
        :param kernel_initializer: initializing method for kernel
        :param recurrent_initializer: initializing method for recurrent kernel
        """
        super(CustomCell, self).__init__()

        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.state_size = self.units

        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool_1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.FC = tf.keras.layers.Dense(512)
        self.kernel = None
        self.recurrent_kernel = None

    def build(self, input_shapes):
        """
        Initialize model when model.build() or model(dummy_input) is called
        :param input_shapes: dummy_input to determine weights size
        """
        # expect input_shape to contain 2 items, [img shape: (None, timestep, patchsize, patch_size, channels,
        #                                         metadata_shape: (None, metadata_len)]
        img_shape = input_shapes[0][1:]
        metadata_shape = input_shapes[1][1]
        img_inp = tf.keras.Input(img_shape)
        x = self.conv_1(img_inp)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        # concatenate encoded image and metadata
        self.kernel = self.add_weight(
            shape=(x.shape[1] + metadata_shape, self.units), initializer=self.kernel_initializer, name="kernel")
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer, name="recurrent_kernel")

    def call(self, inputs, state):
        """
        Called at each loop of the RNN
        :param inputs: image, metadata
        :param state: previous state
        :return: output, hidden state
        """
        img, metadata = inputs
        x = self.conv_1(img)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        # concatenate encoded image and metadatat
        combined = tf.keras.layers.concatenate([x, metadata], 1)
        input_infl = tf.matmul(combined, self.kernel)
        recurrent_infl = tf.matmul(state, self.recurrent_kernel)
        recurrent_infl = tf.reshape(recurrent_infl, (-1, self.units))

        # h i.e. typical hidden layer update
        h = self.activation(input_infl + recurrent_infl)
        o = self.FC(h)
        return o, h


class RNN(tf.keras.Model):
    """
    Basic RNN
    """
    def __init__(self, **kwargs):
        """
        Define model layers
        """
        super(RNN, self).__init__(**kwargs)
        self.rnn = tf.keras.layers.RNN(CustomCell(512), name='rnn')
        self.FC = tf.keras.layers.Dense(4, name='outputs')

    def call(self, inputs):
        """
        Perform forward on inputs and returns predicted GHI at the
        desired times
        :param inputs: input images and metadata
        :return: a list of four floats for GHI at each desired time
        """
        img_seq, metadata_seq, future_metadata = inputs
        x = self.rnn((img_seq, metadata_seq))
        combined = tf.keras.layers.concatenate([x, future_metadata], 1)
        outputs = self.FC(combined)
        return outputs
