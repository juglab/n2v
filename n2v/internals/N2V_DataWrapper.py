from csbdeep.internals.train import RollingSequence
from tensorflow.keras.utils import Sequence

import numpy as np

class N2V_DataWrapper(RollingSequence):
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

    def __init__(self, X, Y, batch_size, length, perc_pix=0.198, shape=(64, 64),
                 value_manipulation=None, structN2Vmask=None):
        super(N2V_DataWrapper, self).__init__(data_size=len(X), batch_size=batch_size, length=length)
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape = shape
        self.value_manipulation = value_manipulation
        self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape)
        self.dims = len(shape)
        self.n_chan = X.shape[-1]
        self.structN2Vmask = structN2Vmask
        if self.structN2Vmask is not None:
            print("StructN2V Mask is: ", self.structN2Vmask)

        num_pix = int(np.product(shape)/100.0 * perc_pix)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(100.0/np.product(shape))
        print("{} blind-spots will be generated per training patch of size {}.".format(num_pix, shape))

        if self.dims == 2:
            self.patch_sampler = self.__subpatch_sampling2D__
            self.box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords2D__
            self.rand_float = self.__rand_float_coords2D__(self.box_size)
        elif self.dims == 3:
            self.patch_sampler = self.__subpatch_sampling3D__
            self.box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords3D__
            self.rand_float = self.__rand_float_coords3D__(self.box_size)
        else:
            raise Exception('Dimensionality not supported.')

        self.X_Batches = np.zeros((self.batch_size, *self.shape, self.n_chan), dtype=np.float32)
        self.Y_Batches = np.zeros((self.batch_size, *self.shape, 2*self.n_chan), dtype=np.float32)

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))


    def __getitem__(self, i):
        idx = self.batch(i)
        # idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        # idx = self.perm[idx]
        self.X_Batches *= 0
        self.Y_Batches *= 0
        self.patch_sampler(self.X, self.X_Batches, indices=idx, range=self.range, shape=self.shape)

        for c in range(self.n_chan):
            for j in range(self.batch_size):
                coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                    shape=self.shape)

                indexing = (j,) + coords + (c,)
                indexing_mask = (j,) + coords + (c + self.n_chan, )
                y_val = self.X_Batches[indexing]
                x_val = self.value_manipulation(self.X_Batches[j, ..., c], coords, self.dims)

                self.Y_Batches[indexing] = y_val
                self.Y_Batches[indexing_mask] = 1
                self.X_Batches[indexing] = x_val
                
                if self.structN2Vmask is not None:
                    self.apply_structN2Vmask(self.X_Batches[j, ..., c], coords, self.dims, self.structN2Vmask)

        return self.X_Batches, self.Y_Batches

    def apply_structN2Vmask(self, patch, coords, dims, mask):
        """
        each point in coords corresponds to the center of the mask.
        then for point in the mask with value=1 we assign a random value
        """
        coords = np.array(coords).astype(np.int)
        ndim = mask.ndim
        center = np.array(mask.shape)//2
        ## leave the center value alone
        mask[tuple(center.T)] = 0
        ## displacements from center
        dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
        ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
        mix = (dx.T[...,None] + coords[None])
        mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
        ## stay within patch boundary
        mix = mix.clip(min=np.zeros(ndim),max=np.array(patch.shape)-1).astype(np.uint)
        ## replace neighbouring pixels with random values from flat dist
        patch[tuple(mix.T)] = np.random.rand(mix.shape[0])*4 - 2

    # return x_val_structN2V, indexing_structN2V
    @staticmethod
    def __subpatch_sampling2D__(X, X_Batches, indices, range, shape):
        for i, j in enumerate(indices):
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_Batches[i] = np.copy(X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]])

    @staticmethod
    def __subpatch_sampling3D__(X, X_Batches, indices, range, shape):
        for i, j in enumerate(indices):
            z_start = np.random.randint(0, range[0] + 1)
            y_start = np.random.randint(0, range[1] + 1)
            x_start = np.random.randint(0, range[2] + 1)
            X_Batches[i] = np.copy(X[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]])

    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    y_coords.append(y)
                    x_coords.append(x)
        return (y_coords, x_coords)

    @staticmethod
    def __get_stratified_coords3D__(coord_gen, box_size, shape):
        box_count_z = int(np.ceil(shape[0] / box_size))
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        z_coords = []
        for i in range(box_count_z):
            for j in range(box_count_y):
                for k in range(box_count_x):
                    z, y, x = next(coord_gen)
                    z = int(i * box_size + z)
                    y = int(j * box_size + y)
                    x = int(k * box_size + x)
                    if (z < shape[0] and y < shape[1] and x < shape[2]):
                        z_coords.append(z)
                        y_coords.append(y)
                        x_coords.append(x)
        return (z_coords, y_coords, x_coords)

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

    @staticmethod
    def __rand_float_coords3D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)