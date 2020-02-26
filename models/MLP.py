"""
Basic MLP model takes metadata from previous time points as input
(no images) to predict GHI at T0, T1, T3 and T6
"""
import tensorflow as tf


class MLP(tf.keras.Model):
    """
    MLP trained and tested on past, present and future metadata
    to serve as baseline
    """
    def __init__(self, hidden=(8, 8)):
        """
        Define model layers
        """
        super(MLP, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(hidden[0], activation=tf.nn.relu,
                                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.005, l2=0.005))
        self.dpout_1 = tf.keras.layers.Dropout(0.1)
        self.dense_2 = tf.keras.layers.Dense(hidden[1], activation=tf.nn.relu,
                                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.005, l2=0.005))
        self.dpout_2 = tf.keras.layers.Dropout(0.1)
        self.out = tf.keras.layers.Dense(4, activation='linear',
                                         name='output')

    def call(self, inputs):
        """
        Perform forward on inputs and returns predicted GHI at t0,
        t1, t3 and t6
        :param inputs: input metadata (past, present, and future)
        :return: a list of four floats for GHI at each desired time
        """
        img, past_metadata, future_metadata = inputs
        # past metadata of shape: n x timesteps x data points (5): day/hour/min/daytime/clearsky
        # future metadata of shape: n x time outputs (4)
        # reshaped past metadata to flatten to 1D, concatenate w future metadata
        past_metadata = tf.reshape(past_metadata, (-1, past_metadata.shape[1]*past_metadata.shape[2]))
        x = tf.keras.layers.concatenate([past_metadata, future_metadata], 1)
        x = self.dense_1(x)
        x = self.dpout_1(x)
        x = self.dense_2(x)
        x = self.dpout_2(x)
        # 4 outputs for t0, t0+1, t0+3 and t0+6
        t_out = self.out(x)

        return t_out
