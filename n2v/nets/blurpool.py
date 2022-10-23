import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras


class MaxBlurPool2D(Layer):
    """
    MaxBlurPool proposed in:
    Zhang, Richard. "Making convolutional networks shift-invariant again."
    International conference on machine learning. PMLR, 2019.

    Implementation inspired by: https://github.com/csvance/blur-pool-keras
    """

    def __init__(self, pool, **kwargs):
        self.pool = pool
        self.blur_kernel = None

        super(MaxBlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):
        gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        gaussian = gaussian / np.sum(gaussian)

        gaussian = np.repeat(gaussian, input_shape[3])

        gaussian = np.reshape(gaussian, (3, 3, input_shape[3], 1))
        blur_init = keras.initializers.constant(gaussian)

        self.blur_kernel = self.add_weight(
            name="blur_kernel",
            shape=(3, 3, input_shape[3], 1),
            initializer=blur_init,
            trainable=False,
        )

        super(MaxBlurPool2D, self).build(input_shape)

    def call(self, x, **kwargs):

        x = tf.nn.pool(
            x,
            (self.pool[0], self.pool[1]),
            strides=(1, 1),
            padding="SAME",
            pooling_type="MAX",
            data_format="NHWC",
        )
        x = K.depthwise_conv2d(x, self.blur_kernel, padding="same",
                               strides=(self.pool[0], self.pool[1]))

        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / 2)),
            int(np.ceil(input_shape[2] / 2)),
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool": self.pool
        })
        return config
