from n2v.internals.N2V_DataWrapper import N2V_DataWrapper

import numpy as np


def test_subpatch_sampling():
    def create_data(in_shape, out_shape):
        X, Y = np.random.rand(*in_shape), np.random.rand(*in_shape)
        X_Batches, Y_Batches = np.zeros(out_shape), np.zeros(out_shape)
        indices = np.arange(in_shape[0])
        np.random.shuffle(indices)

        return X, Y, X_Batches, Y_Batches, indices[:in_shape[0] // 2]

    def _sample2D(in_shape, out_shape, seed):
        X, Y, X_Batches, Y_Batches, indices = create_data(in_shape, out_shape)
        np.random.seed(seed)
        N2V_DataWrapper.__subpatch_sampling2D__(X, X_Batches, indices,
                                                range=in_shape[1:3] - out_shape[1:3], shape=out_shape[1:3])

        assert ([*X_Batches.shape] == out_shape).all()
        np.random.seed(seed)
        range_y = in_shape[1] - out_shape[1]
        range_x = in_shape[2] - out_shape[2]
        for i, j in enumerate(indices):
            assert np.sum(X_Batches[i]) != 0
            y_start = np.random.randint(0, range_y + 1)
            x_start = np.random.randint(0, range_x + 1)
            assert np.sum(X_Batches[i] - X[j, y_start:y_start + out_shape[1], x_start:x_start + out_shape[2]]) == 0

    def _sample3D(in_shape, out_shape, seed):
        X, Y, X_Batches, Y_Batches, indices = create_data(in_shape, out_shape)
        np.random.seed(seed)
        N2V_DataWrapper.__subpatch_sampling3D__(X, X_Batches, indices,
                                                range=in_shape[1:4] - out_shape[1:4], shape=out_shape[1:4])

        assert ([*X_Batches.shape] == out_shape).all()
        np.random.seed(seed)
        range_z = in_shape[1] - out_shape[1]
        range_y = in_shape[2] - out_shape[2]
        range_x = in_shape[3] - out_shape[3]
        for i, j in enumerate(indices):
            assert np.sum(X_Batches[i]) != 0
            z_start = np.random.randint(0, range_z + 1)
            y_start = np.random.randint(0, range_y + 1)
            x_start = np.random.randint(0, range_x + 1)
            assert np.sum(X_Batches[i] - X[j, z_start:z_start + out_shape[1], y_start:y_start + out_shape[2],
                                         x_start:x_start + out_shape[3]]) == 0

    _sample2D(np.array([20, 64, 64, 2]), np.array([20, 32, 32, 2]), 1)
    _sample2D(np.array([10, 25, 25, 1]), np.array([10, 12, 12, 1]), 2)
    _sample2D(np.array([10, 25, 25, 3]), np.array([10, 13, 13, 3]), 3)

    _sample3D(np.array([20, 64, 64, 64, 2]), np.array([20, 32, 32, 32, 2]), 1)
    _sample3D(np.array([10, 25, 25, 25, 1]), np.array([10, 12, 12, 12, 1]), 2)
    _sample3D(np.array([10, 25, 25, 25, 3]), np.array([10, 13, 13, 13, 3]), 3)


def test_random_float_coords():
    boxsize = 13
    np.random.seed(1)
    coords = (np.random.rand() * boxsize, np.random.rand() * boxsize)
    np.random.seed(1)
    assert next(N2V_DataWrapper.__rand_float_coords2D__(boxsize)) == coords

    boxsize = 3
    np.random.seed(1)
    coords = (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)
    np.random.seed(1)
    assert next(N2V_DataWrapper.__rand_float_coords3D__(boxsize)) == coords


def test_coord_gen():
    coord_gen = N2V_DataWrapper.__rand_float_coords2D__(13)
    shape = [128, 128]
    for i in range(100):
        coords = next(coord_gen)
        assert coords[0] < shape[0]
        assert coords[1] < shape[1]

    coord_gen = N2V_DataWrapper.__rand_float_coords3D__(4)
    shape = [55, 55, 55]
    for i in range(100):
        coords = next(coord_gen)
        assert coords[0] < shape[0]
        assert coords[1] < shape[1]
        assert coords[2] < shape[2]


def test_n2vWrapper_getitem():
    def create_data(y_shape):
        Y = np.random.rand(*y_shape)
        return Y

    def random_neighbor_withCP_uniform(patch, coord, dims):
        return np.random.rand(1, 1)

    def _getitem2D(y_shape):
        Y = create_data(y_shape)
        n_chan = y_shape[-1] // 2
        if n_chan == 1:
            X = Y[:, :, :, 0][:, :, :, np.newaxis]
        else:
            X = Y[:, :, :, :n_chan]
        val_manipulator = random_neighbor_withCP_uniform
        dw = N2V_DataWrapper(X, Y, 4, length=16, perc_pix=0.198, shape=(32, 32), value_manipulation=val_manipulator)

        x_batch, y_batch = dw.__getitem__(0)
        assert x_batch.shape == (4, 32, 32, int(n_chan))
        assert y_batch.shape == (4, 32, 32, int(2 * n_chan))
        # At least one pixel has to be a blind-spot per batch sample
        assert np.sum(y_batch[..., n_chan:]) >= 4 * n_chan
        # At most four pixels can be affected per batch sample
        assert np.sum(y_batch[..., n_chan:]) <= 4 * 4 * n_chan

    def _getitem3D(y_shape):
        Y = create_data(y_shape)
        n_chan = y_shape[-1] // 2
        X = Y[:, :, :, :, 0][:, :, :, :, np.newaxis]
        val_manipulator = random_neighbor_withCP_uniform
        dw = N2V_DataWrapper(X, Y, 4, length=16, perc_pix=0.198, shape=(32, 32, 32), value_manipulation=val_manipulator)

        x_batch, y_batch = dw.__getitem__(0)
        assert x_batch.shape == (4, 32, 32, 32, 1)
        assert y_batch.shape == (4, 32, 32, 32, 2)
        # At least one pixel has to be a blind-spot per batch sample
        assert np.sum(y_batch[..., n_chan:]) >= 1 * 4 * n_chan
        # At most 8 pixels can be affected per batch sample
        assert np.sum(y_batch[..., n_chan:]) <= 8 * 4 * n_chan

    _getitem2D(np.array([4, 32, 32, 2]))
    _getitem2D(np.array([4, 64, 64, 2]))
    _getitem2D(np.array([4, 44, 55, 2]))
    _getitem2D(np.array([4, 45, 41, 4]))
    _getitem3D(np.array([4, 32, 32, 32, 2]))
    _getitem3D(np.array([4, 64, 64, 64, 2]))
    _getitem3D(np.array([4, 33, 44, 55, 2]))
