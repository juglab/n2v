from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last

import numpy as np
import keras.backend as K

import tensorflow as tf


def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)


def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[-1]
            mu    = y_pred[...,:n]
            sigma = y_pred[...,n:]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll
    else:
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[1]
            mu    = y_pred[:,:n,...]
            sigma = y_pred[:,n:,...]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll


#def loss_mae(mean=True):
    #R = _mean_or_not(mean)
    #if backend_channels_last():
        #def mae(y_true, y_pred):
            #n = K.shape(y_true)[-1]
            #return R(K.abs(y_pred[...,:n] - y_true))
        #return mae
    #else:
        #def mae(y_true, y_pred):
            #n = K.shape(y_true)[1]
            #return R(K.abs(y_pred[:,:n,...] - y_true))
        #return mae


#def loss_mse(mean=True):
    #R = _mean_or_not(mean)
    #if backend_channels_last():
        #def mse(y_true, y_pred):
            #n = K.shape(y_true)[-1]
            #return R(K.square(y_pred[...,:n] - y_true))
        #return mse
    #else:
        #def mse(y_true, y_pred):
            #n = K.shape(y_true)[1]
            #return R(K.square(y_pred[:,:n,...] - y_true))
        #return mse

def loss_mae(mean=True):
    R = _mean_or_not(mean)
    def mae(y_true, y_pred):
        return R(K.abs(y_pred[...,0] - y_true[...,0]))
    return mae

def loss_mse(mean=True):
    R = _mean_or_not(mean)
    def mse(y_true, y_pred):
        return R(K.square(y_pred[...,0] - y_true[...,0]))
    return mse


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss

def loss_noise2void():
    def n2v_mse(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        loss = tf.reduce_sum(K.square(target - y_pred*mask)) / tf.reduce_sum(mask)
        return loss

    return n2v_mse

def loss_noise2voidAbs():
    def n2v_abs(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        loss = tf.reduce_sum(K.abs(target - y_pred*mask)) / tf.reduce_sum(mask)
        return loss

    return n2v_abs
