import numpy as np
from ..internals.train import  Noise2VoidDataWrapper as dw


def get_subpatch(patch, coord, local_sub_patch_radius):
    start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1

    shift = np.minimum(0, patch.shape - end)

    start += shift
    end += shift

    slices = [ slice(s, e) for s, e in zip(start, end)]

    return patch[slices]


def random_neighbor(shape, coord):
    rand_coords = sample_coords(shape, coord)
    while np.any(rand_coords == coord):
        rand_coords = sample_coords(shape, coord)

    return rand_coords


def sample_coords(shape, coord, sigma=4):
    return [normal_int(c, sigma, s) for c, s in zip(coord, shape)]


def normal_int(mean, sigma, w):
    return int(np.clip(np.round(np.random.normal(mean, sigma)), 0, w - 1))


def pm_normal_withoutCP(patch, coord):
    rand_coords = random_neighbor(patch.shape, coord)
    return patch[tuple(rand_coords)]


def pm_uniform_withCP(local_sub_patch_radius):
    def random_neighbor_withCP_uniform(patch, coord):
        sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
        rand_coords = [np.random.randint(0, s) for s in sub_patch.shape]
        return sub_patch[tuple(rand_coords)]
    return random_neighbor_withCP_uniform


def pm_normal_additive(pixel_gauss_sigma):
    def pixel_gauss(patch, coord):
        return np.random.normal(patch[tuple(coord)], pixel_gauss_sigma)
    return pixel_gauss


def pm_normal_fitted(local_sub_patch_radius):
    def local_gaussian(patch, coord):
        sub_patch = get_subpatch(patch, coord, local_sub_patch_radius)
        return np.random.normal(np.mean(sub_patch), np.std(sub_patch))

    return local_gaussian


def pm_identity(patch, coord):
    return patch[tuple(coord)]


def manipulate_val_data(X_val, Y_val, num_pix=64, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    if len(shape) == 2:
        box_size = np.round(np.sqrt(shape[0] * shape[1] / num_pix)).astype(np.int)
        get_stratified_coords = dw.__get_stratified_coords2D__
        rand_float = dw.__rand_float_coords2D__(box_size)
    elif len(shape) == 3:
        box_size = np.round(np.power(shape[0] * shape[1] * shape[2] / num_pix, 1 / 3.0)).astype(np.int)
        get_stratified_coords = dw.__get_stratified_coords3D__
        rand_float = dw.__rand_float_coords3D__(box_size)

    for j in range(X_val.shape[0]):
        coords = get_stratified_coords(rand_float, box_size=box_size,
                                            shape=np.array(X_val.shape)[1:-1])
        y_val = []
        x_val = []
        for k in range(len(coords)):
            y_val.append(Y_val[(j, *coords[k], 0)])
            x_val.append(value_manipulation(X_val[j, ..., 0], coords[k]))

        Y_val[j] *= 0

        for k in range(len(coords)):
            Y_val[(j, *coords[k], 0)] = y_val[k]
            Y_val[(j, *coords[k], 1)] = 1

            X_val[(j, *coords[k])] = x_val[k]
