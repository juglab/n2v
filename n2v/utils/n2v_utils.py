import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from ..internals.N2V_DataWrapper import N2V_DataWrapper as dw


def get_subpatch(patch, coord, local_sub_patch_radius, crop_patch=True):
    crop_neg, crop_pos = 0, 0
    if crop_patch:
        start = np.array(coord) - local_sub_patch_radius
        end = start + local_sub_patch_radius * 2 + 1

        # compute offsets left/up ...
        crop_neg = np.minimum(start, 0)
        # and right/down
        crop_pos = np.maximum(0, end-patch.shape)

        # correct for offsets, patch size shrinks if crop_*!=0
        start -= crop_neg
        end -= crop_pos
    else:
        start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
        end = start + local_sub_patch_radius * 2 + 1

        shift = np.minimum(0, patch.shape - end)

        start += shift
        end += shift

    slices = [slice(s, e) for s, e in zip(start, end)]

    # return crop vectors for deriving correct center pixel locations later
    return patch[tuple(slices)], crop_neg, crop_pos


def random_neighbor(shape, coord):
    rand_coords = sample_coords(shape, coord)
    while np.any(rand_coords == coord):
        rand_coords = sample_coords(shape, coord)

    return rand_coords


def sample_coords(shape, coord, sigma=4):
    return [normal_int(c, sigma, s) for c, s in zip(coord, shape)]


def normal_int(mean, sigma, w):
    return int(np.clip(np.round(np.random.normal(mean, sigma)), 0, w - 1))


def mask_center(local_sub_patch_radius, ndims=2):
    size = local_sub_patch_radius*2 + 1
    patch_wo_center = np.ones((size, ) * ndims)
    if ndims == 2:
        patch_wo_center[local_sub_patch_radius, local_sub_patch_radius] = 0
    elif ndims == 3:
        patch_wo_center[local_sub_patch_radius,
        local_sub_patch_radius, local_sub_patch_radius] = 0
    else:
        raise NotImplementedError()
    return ma.make_mask(patch_wo_center)


def pm_normal_withoutCP(local_sub_patch_radius):
    def normal_withoutCP(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            rand_coords = random_neighbor(patch.shape, coord)
            vals.append(patch[tuple(rand_coords)])
        return vals

    return normal_withoutCP


def pm_mean(local_sub_patch_radius):
    def patch_mean(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.mean(sub_patch[sub_patch_mask]))
        return vals

    return patch_mean


def pm_median(local_sub_patch_radius):
    def patch_median(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.median(sub_patch[sub_patch_mask]))
        return vals

    return patch_median


def pm_uniform_withCP(local_sub_patch_radius):
    def random_neighbor_withCP_uniform(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            sub_patch, _, _ = get_subpatch(patch, coord, local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals

    return random_neighbor_withCP_uniform


def pm_uniform_withoutCP(local_sub_patch_radius):
    def random_neighbor_withoutCP_uniform(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.random.permutation(sub_patch[sub_patch_mask])[0])
        return vals

    return random_neighbor_withoutCP_uniform


def pm_normal_additive(pixel_gauss_sigma):
    def pixel_gauss(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            vals.append(np.random.normal(patch[tuple(coord)], pixel_gauss_sigma))
        return vals

    return pixel_gauss


def pm_normal_fitted(local_sub_patch_radius):
    def local_gaussian(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            sub_patch, _, _ = get_subpatch(patch, coord, local_sub_patch_radius)
            axis = tuple(range(dims))
            vals.append(np.random.normal(np.mean(sub_patch, axis=axis), np.std(sub_patch, axis=axis)))
        return vals

    return local_gaussian


def pm_identity(local_sub_patch_radius):
    def identity(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            vals.append(patch[coord])
        return vals

    return identity


def manipulate_val_data(X_val, Y_val, perc_pix=0.198, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int32)
        get_stratified_coords = dw.__get_stratified_coords2D__
        rand_float = dw.__rand_float_coords2D__(box_size)
    elif dims == 3:
        box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int32)
        get_stratified_coords = dw.__get_stratified_coords3D__
        rand_float = dw.__rand_float_coords3D__(box_size)

    n_chan = X_val.shape[-1]

    Y_val *= 0
    for j in tqdm(range(X_val.shape[0]), desc='Preparing validation data: '):
        coords = get_stratified_coords(rand_float, box_size=box_size,
                                       shape=np.array(X_val.shape)[1:-1])
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)
            y_val = X_val[indexing]
            x_val = value_manipulation(X_val[j, ..., c], coords, dims)

            Y_val[indexing] = y_val
            Y_val[indexing_mask] = 1
            X_val[indexing] = x_val


def autocorrelation(x):
    """
    nD autocorrelation
    remove mean per-patch (not global GT)
    normalize stddev to 1
    value at zero shift normalized to 1...
    """
    x = (x - np.mean(x)) / np.std(x)
    x = np.fft.fftn(x)
    x = np.abs(x) ** 2
    x = np.fft.ifftn(x).real
    x = x / x.flat[0]
    x = np.fft.fftshift(x)
    return x


def tta_forward(x):
    """
    Augments x 8-fold: all 90 deg rotations plus lr flip of the four rotated versions.

    Parameters
    ----------
    x: data to augment

    Returns
    -------
    Stack of augmented x.
    """
    x_aug = [x, np.rot90(x, 1), np.rot90(x, 2), np.rot90(x, 3)]
    x_aug_flip = x_aug.copy()
    for x_ in x_aug:
        x_aug_flip.append(np.fliplr(x_))
    return x_aug_flip


def tta_backward(x_aug):
    """
    Inverts `tta_forward` and averages the 8 images.

    Parameters
    ----------
    x_aug: stack of 8-fold augmented images.

    Returns
    -------
    average of de-augmented x_aug.
    """
    x_deaug = [x_aug[0], np.rot90(x_aug[1], -1), np.rot90(x_aug[2], -2), np.rot90(x_aug[3], -3),
               np.fliplr(x_aug[4]), np.rot90(np.fliplr(x_aug[5]), -1), np.rot90(np.fliplr(x_aug[6]), -2), np.rot90(np.fliplr(x_aug[7]), -3)]
    return np.mean(x_deaug, 0)
