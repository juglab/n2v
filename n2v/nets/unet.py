from __future__ import print_function, unicode_literals, absolute_import, division

from tensorflow.keras.layers import Input, Conv2D, Conv3D, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Concatenate
from csbdeep.internals.blocks import unet_block

import tensorflow as tf

from csbdeep.utils.utils import _raise, backend_channels_last
import numpy as np


def build_single_unet_per_channel(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                pool_size=(2,2,2),
                residual=False,
                prob_out=False,
                eps_scale=1e-3):
    """ TODO """

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 if backend_channels_last() else 1

    n_dim = len(kernel_size)
    conv = Conv2D if n_dim==2 else Conv3D

    num_channels = input_shape[channel_axis]

    input = Input(input_shape, name = "input")

    out_channels = []
    num_channel_out = 1
    for i in range(num_channels):
        c = Lambda(lambda x: x[:, ..., i:i+1])(input)
        unet = unet_block(n_depth, n_filter_base, kernel_size,
                          activation=activation, dropout=dropout, batch_norm=batch_norm,
                          n_conv_per_depth=n_conv_per_depth, pool=pool_size, prefix='channel_{}'.format(i))(c)

        final = conv(num_channel_out, (1,)*n_dim, activation='linear')(unet)
        if residual:
            if not (num_channel_out == 1 if backend_channels_last() else num_channel_out == 1):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, input])
        final = Activation(activation=last_activation)(final)

        if prob_out:
            scale = conv(num_channel_out, (1,)*n_dim, activation='softplus')(unet)
            scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
            final = Concatenate(axis=channel_axis)([final,scale])

        out_channels.append(final)

    if len(out_channels) > 1:
        output = Concatenate(axis=channel_axis)(out_channels)
        return Model(inputs=input, outputs=output)
    else:
        return Model(inputs=input, outputs=out_channels[0])

