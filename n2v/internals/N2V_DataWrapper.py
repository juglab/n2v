from keras.utils import Sequence

import numpy as np


class N2V_DataWrapper(Sequence):
    """
    The N2V_DataWrapper extracts random sub-patches from the given data and manipulates 'num_pix' pixels in the
    input.

    Parameters
    ----------
    X          : array(floats)
                 The noisy input data. ('SZYXC' or 'SYXC')
    Y          : array(floats)
                 The same as X plus a masking channel.
    batch_size : int
                 Number of samples per batch.
    num_pix    : int, optional(default=1)
                 Number of pixels to manipulate.
    shape      : tuple(int), optional(default=(64, 64))
                 Shape of the randomly extracted patches.
    value_manipulator : function, optional(default=None)
                        The manipulator used for the pixel replacement.
    """

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
            for c in range(self.n_chan):
                coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                    shape=np.array(self.X_Batches.shape)[1:-1])

                y_val = []
                x_val = []
                for k in range(len(coords)):
                    y_val.append(np.copy(self.Y_Batches[(j, *coords[k], ..., c)]))
                    x_val.append(self.value_manipulation(self.X_Batches[j, ..., c][...,np.newaxis], coords[k], self.dims))

                self.Y_Batches[j,...,c] *= 0
                self.Y_Batches[j,...,self.n_chan+c] *= 0

                for k in range(len(coords)):
                    self.Y_Batches[(j, *coords[k], c)] = y_val[k]
                    self.Y_Batches[(j, *coords[k], self.n_chan+c)] = 1
                    self.X_Batches[(j, *coords[k], c)] = x_val[k]


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