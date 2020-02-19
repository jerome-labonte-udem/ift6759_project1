"""
ConvLSTM model from Shi et al. 2015, Keras implementation
Takes satellite images and metadata from  (5) previous time points
as input to predict GHI at T0, T1, T3 and T6

Doc & References:
https://keras.io/examples/conv_lstm/
https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
https://keras.io/examples/imdb_lstm/
https://www.tensorflow.org/api_docs/python/tf/keras/Model
https://github.com/iwyoo/ConvLSTMCell-tensorflow/blob/master/ConvLSTMCell.py
https://researchcode.com/code/1780809736/convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting/
https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py

"""
import tensorflow as tf


class ConvLSTM(tf.keras.Model):
    """
    ConvLSTM with keras ConvLSTM layers predicts GHI at t0, t1, t3, t6 from
    sequence of images and metadata from previous time stamps,
    and current & future clearsky predictions
    """
    def __init__(self):
        """
        Define model layers
        """
        super(ConvLSTM, self).__init__()
        # Number of filters approximated from Shi 2015 (2 conv layers of 64x64)
        # Note that the 2nd and 3rd values of input_shape = patch size
        self.convlstm_1 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                                     input_shape=(None, 64, 64, 5),
                                                     padding='same',
                                                     return_sequences=True)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.convlstm_2 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                                     input_shape=(None, 64, 64, 5),
                                                     padding='same',
                                                     return_sequences=True)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Option to add 3rd conv_lstm layer (shit 2015 got best results w 2 and 3)
        # self.convlstm_3 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
        #           input_shape=(None, 64, 64, 5), padding='same',
        #           return_sequences=True)
        # self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv3D = tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3),
                                             activation='sigmoid', padding='same',
                                             data_format='channels_last')

        self.flatten = tf.keras.layers.Flatten()

        # LSTM for previous metadata (2 layers; 1 layer also works well)
        self.lstm_1 = tf.keras.layers.LSTM(8, return_sequences=True, dropout=0.1,
                                           recurrent_dropout=0.1)
        # Note: return_sequences = False by default, layer outputs 2D rather than
        # 3D data (can concatenate with future metadata vector)
        self.lstm_2 = tf.keras.layers.LSTM(8, dropout=0.1, recurrent_dropout=0.1)

        self.FC_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu,
                                          kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.005, l2=0.005))
        self.FC_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu,
                                          kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.005, l2=0.005))
        self.dpout_1 = tf.keras.layers.Dropout(0.1)
        self.dpout_2 = tf.keras.layers.Dropout(0.1)

        self.out = tf.keras.layers.Dense(4, activation='linear',
                                         name='output')

    def call(self, inputs):
        """
        Perform forward on inputs and returns predicted GHI at t0,
        t1, t3 and t6
        :param inputs: input images and metadata (past, present, and future)
        :return: a list of four floats for GHI at each desired time
        """
        image_seq, past_metadata, future_metadata = inputs
        # image_seq of shape: n x timesteps x patch_size x patch_size x channels
        # past metadata of shape: n x timesteps x data points (5: day, hour, min, day-night, clear sky)
        # future metadata of shape: n x time outputs (4)
        # patch_size = image_seq.shape[2]
        # n_channels = image_seq.shape[4]

        img = self.convlstm_1(image_seq)
        img = self.bn1(img)
        img = self.convlstm_2(img)
        img = self.bn2(img)
        # img = self.convlstm_3(img)
        # img = self.bn3(img)
        img = self.conv3D(img)
        img = self.flatten(img)

        pmd = self.lstm_1(past_metadata)
        pmd = self.lstm_2(pmd)
        # no need to reshape past_metadata w use of recurrence: output is flattened
        # pmd = tf.reshape(pmd, (-1, past_metadata.shape[1]*past_metadata.shape[2]))

        merged = tf.keras.layers.concatenate([img, pmd, future_metadata], 1)
        merged = self.FC_1(merged)
        merged = self.dpout_1(merged)
        merged = self.FC_2(merged)
        merged = self.dpout_2(merged)
        # 4 outputs for t0, t0+1, t0+3 and t0+6
        outputs = self.out(merged)

        return outputs
