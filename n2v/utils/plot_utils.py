from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np

from .utils import normalize



def plot_history(history,*keys,**kwargs):
    """Plot (Keras) training history returned by :func:`CARE.train`."""
    import matplotlib.pyplot as plt

    logy = kwargs.pop('logy',False)

    if all(( isinstance(k,string_types) for k in keys )):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1,w,i+1)
        for k in ([group] if isinstance(group,string_types) else group):
            plt.plot(history.epoch,history.history[k],'.-',label=k,**kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')
    # plt.tight_layout()
    plt.show()


def plot_some(*arr, **kwargs):
    """Quickly plot multiple images at once."""

    title_list = kwargs.pop('title_list',None)
    pmin = kwargs.pop('pmin',0)
    pmax = kwargs.pop('pmax',100)
    cmap = kwargs.pop('cmap','magma')
    imshow_kwargs = kwargs
    return _plot_some(arr=arr, title_list=title_list, pmin=pmin, pmax=pmax, cmap=cmap, **imshow_kwargs)

def _plot_some(arr, title_list=None, pmin=0, pmax=100, cmap='magma', **imshow_kwargs):
    """
    plots a matrix of images

    arr = [ X_1, X_2, ..., X_n]

    where each X_i is a list of images

    :param arr:
    :param title_list:
    :param pmin:
    :param pmax:
    :param imshow_kwargs:
    :return:
    """
    import matplotlib.pyplot as plt

    imshow_kwargs['cmap'] = cmap

    def color_image(a):
        return np.stack(map(to_color,a)) if 1<a.shape[-1]<=3 else np.squeeze(a)
    def max_project(a):
        return np.max(a,axis=1) if (a.ndim==4 and not 1<=a.shape[-1]<=3) else a

    arr = map(color_image,arr)
    arr = map(max_project,arr)
    arr = list(arr)

    h = len(arr)
    w = len(arr[0])
    plt.gcf()
    for i in range(h):
        for j in range(w):
            plt.subplot(h, w, i * w + j + 1)
            try:
                plt.title(title_list[i][j], fontsize=8)
            except:
                pass
            img = arr[i][j]
            if pmin!=0 or pmax!=100:
                img = normalize(img,pmin=pmin,pmax=pmax,clip=True)
            plt.imshow(np.squeeze(img),**imshow_kwargs)
            plt.axis("off")


def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2,3):
        raise ValueError("only 2d or 3d arrays supported")

    if arr.ndim ==2:
        arr = arr[np.newaxis]

    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)

    out = np.zeros(arr.shape[1:] + (3,))

    eps = 1.e-20
    if pmin>=0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0

    if pmax>=0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1.+eps

    arr_norm = (1. * arr - mi) / (ma - mi + eps)


    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]

    return np.clip(out, 0, 1)