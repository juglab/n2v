from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, move_channel_for_backend, axes_dict, axes_check_and_normalize, backend_channels_last
from ..internals.losses import loss_laplace, loss_mse, loss_mae, loss_thresh_weighted_decay, loss_noise2void, \
    loss_noise2voidAbs

import numpy as np


import keras.backend as K
from keras.callbacks import Callback, TerminateOnNaN
from keras.utils import Sequence


class ParameterDecayCallback(Callback):
    """ TODO """
    def __init__(self, parameter, decay, name=None, verbose=0):
        self.parameter = parameter
        self.decay = decay
        self.name = name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        old_val = K.get_value(self.parameter)
        if self.name:
            logs = logs or {}
            logs[self.name] = old_val
        new_val = old_val * (1. / (1. + self.decay * (epoch + 1)))
        K.set_value(self.parameter, new_val)
        if self.verbose:
            print("\n[ParameterDecayCallback] new %s: %s\n" % (self.name if self.name else 'parameter', new_val))


def prepare_model(model, optimizer, loss, training_scheme='CARE', metrics=('mse','mae'),
                  loss_bg_thresh=0, loss_bg_decay=0.06, Y=None):
    """ TODO """

    from keras.optimizers import Optimizer
    isinstance(optimizer,Optimizer) or _raise(ValueError())

    if training_scheme == 'CARE':
        loss_standard   = eval('loss_%s()'%loss)
    elif training_scheme == 'Noise2Void':
        if loss == 'mse':
            loss_standard = eval('loss_noise2void()')
        elif loss == 'mae':
            loss_standard = eval('loss_noise2voidAbs()')

    _metrics = [eval('loss_%s()' % m) for m in metrics]
    callbacks       = [TerminateOnNaN()]

    # checks
    assert 0 <= loss_bg_thresh <= 1
    assert loss_bg_thresh == 0 or Y is not None
    if loss == 'laplace':
        assert K.image_data_format() == "channels_last", "TODO"
        assert model.output.shape.as_list()[-1] >= 2 and model.output.shape.as_list()[-1] % 2 == 0

    # loss
    if loss_bg_thresh == 0:
        _loss = loss_standard
    else:
        freq = np.mean(Y > loss_bg_thresh)
        # print("class frequency:", freq)
        alpha = K.variable(1.0)
        loss_per_pixel = eval('loss_{loss}(mean=False)'.format(loss=loss))
        _loss = loss_thresh_weighted_decay(loss_per_pixel, loss_bg_thresh,
                                           0.5 / (0.1 + (1 - freq)),
                                           0.5 / (0.1 +      freq),
                                           alpha)
        callbacks.append(ParameterDecayCallback(alpha, loss_bg_decay, name='alpha'))
        if not loss in metrics:
            _metrics.append(loss_standard)


    # compile model
    model.compile(optimizer=optimizer, loss=_loss, metrics=_metrics)

    return callbacks


class DataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        return self.X[idx], self.Y[idx]


class Noise2VoidDataWrapper(Sequence):

    def __init__(self, X, Y, batch_size, num_pix=1, shape=(64, 64),
                 value_manipulation=None):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape = shape
        self.value_manipulation = value_manipulation
        self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape)
        self.dims = len(shape)
        self.n_chan = X.shape[-1]

        if self.dims == 2:
            self.patch_sampler = self.__subpatch_sampling2D__
            self.box_size = np.round(np.sqrt(shape[0] * shape[1] / num_pix)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords2D__
            self.rand_float = self.__rand_float_coords2D__(self.box_size)
            self.X_Batches = np.zeros([X.shape[0], shape[0], shape[1], X.shape[3]])
            self.Y_Batches = np.zeros([Y.shape[0], shape[0], shape[1], Y.shape[3]])
        elif self.dims == 3:
            self.patch_sampler = self.__subpatch_sampling3D__
            self.box_size = np.round(np.power(shape[0] * shape[1] * shape[2] / num_pix, 1/3.0)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords3D__
            self.rand_float = self.__rand_float_coords3D__(self.box_size)
            self.X_Batches = np.zeros([X.shape[0], shape[0], shape[1], shape[2], X.shape[4]])
            self.Y_Batches = np.zeros([Y.shape[0], shape[0], shape[1], shape[2], Y.shape[4]])
        else:
            raise Exception('Dimensionality not supported.')

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]
        self.patch_sampler(self.X, self.Y, self.X_Batches, self.Y_Batches, idx, self.range, self.shape)

        for j in idx:
            coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                shape=np.array(self.X_Batches.shape)[1:-1])

            y_val = []
            x_val = []
            for k in range(len(coords)):
                y_val.append(np.copy(self.Y_Batches[(j, *coords[k], ...)]))
                x_val.append(self.value_manipulation(self.X_Batches[j, ...], coords[k], self.dims))

            self.Y_Batches[j] *= 0

            for k in range(len(coords)):
                for c in range(self.n_chan):
                    self.Y_Batches[(j, *coords[k], c)] = y_val[k][c]
                    self.Y_Batches[(j, *coords[k], self.n_chan+c)] = 1
                    self.X_Batches[(j, *coords[k], c)] = x_val[k][c]


        return self.X_Batches[idx], self.Y_Batches[idx]

    @staticmethod
    def __subpatch_sampling2D__(X, Y, X_Batches, Y_Batches, indices, range, shape):
        for j in indices:
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_Batches[j] = X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]]
            Y_Batches[j] = Y[j, y_start:y_start + shape[0], x_start:x_start + shape[1]]

    @staticmethod
    def __subpatch_sampling3D__(X, Y, X_Batches, Y_Batches, indices, range, shape):
        for j in indices:
            z_start = np.random.randint(0, range[0] + 1)
            y_start = np.random.randint(0, range[1] + 1)
            x_start = np.random.randint(0, range[2] + 1)
            X_Batches[j] = X[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]]
            Y_Batches[j] = Y[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]]

    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        coords = []
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    coords.append((y, x))
        return coords

    @staticmethod
    def __get_stratified_coords3D__(coord_gen, box_size, shape):
        coords = []
        box_count_z = int(np.ceil(shape[0] / box_size))
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        for i in range(box_count_z):
            for j in range(box_count_y):
                for k in range(box_count_x):
                    z, y, x = next(coord_gen)
                    z = int(i * box_size + z)
                    y = int(j * box_size + y)
                    x = int(k * box_size + x)
                    if (z < shape[0] and y < shape[1] and x < shape[2]):
                        coords.append((z, y, x))
        return coords

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

    @staticmethod
    def __rand_float_coords3D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)
