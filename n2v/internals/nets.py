from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from keras.layers import Input, Conv2D, Conv3D, Activation, Lambda
from keras.models import Model
from keras.layers.merge import Add, Concatenate
from .blocks import unet_block
import re

from ..utils import _raise, backend_channels_last
import numpy as np


def custom_unet(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                pool_size=(2,2,2),
                n_channel_out=1,
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

    input = Input(input_shape, name = "input")
    unet = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet)
    if residual:
        if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
            raise ValueError("number of input and output channels must be the same for a residual net.")
        final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final,scale])

    return Model(inputs=input, outputs=final)



def common_unet(n_dim=2, n_depth=1, kern_size=3, n_first=16, n_channel_out=1, residual=True, prob_out=False, last_activation='linear', batch_norm=False):
    """Construct a common CARE neural net based on U-Net [1]_ and residual learning [2]_ to be used for image restoration/enhancement.

    Parameters
    ----------
    n_dim : int
        number of image dimensions (2 or 3)
    n_depth : int
        number of resolution levels of U-Net architecture
    kern_size : int
        size of convolution filter in all image dimensions
    n_first : int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out : int
        number of channels of the predicted output image
    residual : bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal
    prob_out : bool
        standard regression (False) or probabilistic prediction (True)
        if True, model will predict two values for each input pixel (mean and positive scale value)
    last_activation : str
        name of activation function for the final output layer
    batch_norm : bool
        activate batch_norm

    Returns
    -------
    function
        Function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
    """
    def _build_this(input_shape):
        return custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,)*n_dim, pool_size=(2,)*n_dim, n_channel_out=n_channel_out, residual=residual, prob_out=prob_out, batch_norm=batch_norm)
    return _build_this



modelname = re.compile("^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?(_(?P<last_activation>.+)-last)?$")
def common_unet_by_name(model):
    r"""Shorthand notation for equivalent use of :func:`common_unet`.

    Parameters
    ----------
    model : str
        define model to be created via string, which is parsed as a regular expression:
        `^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?(_(?P<last_activation>.+)-last)?$`

    Returns
    -------
    function
        Calls :func:`common_unet` with the respective parameters.

    Raises
    ------
    ValueError
        If argument `model` is not a valid string according to the regular expression.

    Example
    -------
    >>> model = common_unet_by_name('resunet2_1_3_16_1out')(input_shape)
    >>> # equivalent to: model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    Todo
    ----
    Backslashes in docstring for regexp not rendered correctly.

    """
    m = modelname.fullmatch(model)
    if m is None:
        raise ValueError("model name '%s' unknown, must follow pattern '%s'" % (model, modelname.pattern))
    # from pprint import pprint
    # pprint(m.groupdict())
    options = {k:int(m.group(k)) for k in ['n_depth','n_first','kern_size']}
    options['prob_out'] = m.group('prob_out') is not None
    options['residual'] = {'unet': False, 'resunet': True}[m.group('model')]
    options['n_dim'] = int(m.group('n_dim'))
    options['n_channel_out'] = 1 if m.group('n_channel_out') is None else int(m.group('n_channel_out'))
    if m.group('last_activation') is not None:
        options['last_activation'] = m.group('last_activation')

    return common_unet(**options)



def receptive_field_unet(n_depth, kern_size, n_dim=2, img_size=1024):
    """Receptive field for U-Net model (pre/post for each dimension)."""
    x = np.zeros((1,)+(img_size,)*n_dim+(1,))
    mid = tuple([s//2 for s in x.shape[1:-1]])
    x[(slice(None),) + mid + (slice(None),)] = 1
    model = common_unet(n_dim=n_dim, n_depth=n_depth, kern_size=kern_size, n_first=8)(x.shape[1:])
    y  = model.predict(x)[0,...,0]
    y0 = model.predict(0*x)[0,...,0]
    ind = np.where(np.abs(y-y0)>0)
    return [(m-np.min(i),np.max(i)-m) for (m,i) in zip(mid,ind)]
