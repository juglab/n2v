# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np
from collections import namedtuple
import sys, os, warnings

from ..utils import _raise, consume, axes_check_and_normalize, axes_dict, move_image_axes



class Transform(namedtuple('Transform',('name','generator','size'))):
    """Extension of :func:`collections.namedtuple` with three fields: `name`, `generator`, and `size`.

    Parameters
    ----------
    name : str
        Name of the applied transformation.
    generator : function
        Function that takes a generator as input and itself returns a generator; input and returned
        generator have the same structure as that of :class:`RawData`.
        The purpose of the returned generator is to augment the images provided by the input generator
        through additional transformations.
        It is important that the returned generator also includes every input tuple unchanged.
    size : int
        Number of transformations applied to every image (obtained from the input generator).
    """

    @staticmethod
    def identity():
        """
        Returns
        -------
        Transform
            Identity transformation that passes every input through unchanged.
        """
        def _gen(inputs):
            for d in inputs:
                yield d
        return Transform('Identity', _gen, 1)

    # def flip(axis):
    #     """TODO"""
    #     def _gen(inputs):
    #         for x,y,m_in in inputs:
    #             axis < x.ndim or _raise(ValueError())
    #             yield x, y, m_in
    #             yield np.flip(x,axis), np.flip(y,axis), None if m_in is None else np.flip(m_in,axis)
    #     return Transform('Flip (axis=%d)'%axis, _gen, 2)



def anisotropic_distortions(
        subsample,
        psf,
        psf_axes       = None,
        poisson_noise  = False,
        gauss_sigma    = 0,
        subsample_axis = 'X',
        yield_target   = 'source',
        crop_threshold = 0.2,
    ):
    """Simulate anisotropic distortions.

    Modify the first image (obtained from input generator) along one axis to mimic the
    distortions that typically occur due to low resolution along the Z axis.
    Note that the modified image is finally upscaled to obtain the same resolution
    as the unmodified input image and is yielded as the 'source' image (see :class:`RawData`).
    The mask from the input generator is simply passed through.

    The following operations are applied to the image (in order):

    1. Convolution with PSF
    2. Poisson noise
    3. Gaussian noise
    4. Subsampling along ``subsample_axis``
    5. Upsampling along ``subsample_axis`` (to former size).


    Parameters
    ----------
    subsample : float
        Subsampling factor to mimic distortions along Z.
    psf : :class:`numpy.ndarray` or None
        Point spread function (PSF) that is supposed to mimic blurring
        of the microscope due to reduced axial resolution. Set to ``None`` to disable.
    psf_axes : str or None
        Axes of the PSF. If ``None``, psf axes are assumed to be the same as of the image
        that it is applied to.
    poisson_noise : bool
        Flag to indicate whether Poisson noise should be applied to the image.
    gauss_sigma : float
        Standard deviation of white Gaussian noise to be added to the image.
    subsample_axis : str
        Subsampling image axis (default X).
    yield_target : str
        Which image from the input generator should be yielded by the generator ('source' or 'target').
        If 'source', the unmodified input/source image (from which the distorted image is computed)
        is yielded as the target image. If 'target', the target image from the input generator is simply
        passed through.
    crop_threshold : float
        The subsample factor must evenly divide the image size along the subsampling axis to prevent
        potential image misalignment. If this is not the case the subsample factors are
        modified and the raw image may be cropped along the subsampling axis
        up to a fraction indicated by `crop_threshold`.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object intended to be used with :func:`create_patches`.

    Raises
    ------
    ValueError
        Various reasons.

    """
    zoom_order = 1

    (np.isscalar(subsample) and subsample >= 1) or _raise(ValueError('subsample must be >= 1'))
    _subsample = subsample

    subsample_axis = axes_check_and_normalize(subsample_axis)
    len(subsample_axis)==1 or _raise(ValueError())

    psf is None or isinstance(psf,np.ndarray) or _raise(ValueError())
    if psf_axes is not None:
        psf_axes = axes_check_and_normalize(psf_axes)

    0 < crop_threshold < 1 or _raise(ValueError())

    yield_target in ('source','target') or _raise(ValueError())

    if psf is None and yield_target == 'source':
        warnings.warn(
            "It is strongly recommended to use an appropriate PSF to "
            "mimic the optical effects of the microscope. "
            "We found that training with synthesized anisotropic images "
            "that were created without a PSF "
            "can sometimes lead to unwanted artifacts in the reconstructed images."
        )


    def _make_normalize_data(axes_in):
        """Move X to front of image."""
        axes_in  = axes_check_and_normalize(axes_in)
        axes_out = subsample_axis
        # (a in axes_in for a in 'XY') or _raise(ValueError('X and/or Y axis missing.'))
        # add axis in axes_in to axes_out (if it doesn't exist there)
        axes_out += ''.join(a for a in axes_in if a not in axes_out)

        def _normalize_data(data,undo=False):
            if undo:
                return move_image_axes(data, axes_out, axes_in)
            else:
                return move_image_axes(data, axes_in, axes_out)
        return _normalize_data


    def _scale_down_up(data,subsample):
        from scipy.ndimage.interpolation import zoom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            factor = np.ones(data.ndim)
            factor[0] = subsample
            return zoom(zoom(data, 1/factor, order=0),
                                     factor, order=zoom_order)


    def _adjust_subsample(d,s,c):
        """length d, subsample s, tolerated crop loss fraction c"""
        from fractions import Fraction

        def crop_size(n_digits,frac):
            _s = round(s,n_digits)
            _div = frac.denominator
            s_multiple_max = np.floor(d/_s)
            s_multiple = (s_multiple_max//_div)*_div
            # print(n_digits, _s,_div,s_multiple)
            size = s_multiple * _s
            assert np.allclose(size,round(size))
            return size

        def decimals(v,n_digits=None):
            if n_digits is not None:
                v = round(v,n_digits)
            s = str(v)
            assert '.' in s
            decimals = s[1+s.find('.'):]
            return int(decimals), len(decimals)

        s = float(s)
        dec, n_digits = decimals(s)
        frac = Fraction(dec,10**n_digits)
        # a multiple of s that is also an integer number must be
        # divisible by the denominator of the fraction that represents the decimal points

        # round off decimals points if needed
        while n_digits > 0 and (d-crop_size(n_digits,frac))/d > c:
            n_digits -= 1
            frac = Fraction(decimals(s,n_digits)[0], 10**n_digits)

        size = crop_size(n_digits,frac)
        if size == 0 or (d-size)/d > c:
            raise ValueError("subsample factor %g too large (crop_threshold=%g)" % (s,c))

        return round(s,n_digits), int(round(crop_size(n_digits,frac)))


    def _make_divisible_by_subsample(x,size):
        def _split_slice(v):
            return slice(None) if v==0 else slice(v//2,-(v-v//2))
        slices = [slice(None) for _ in x.shape]
        slices[0] = _split_slice(x.shape[0]-size)
        return x[slices]


    def _generator(inputs):
        for img,y,axes,mask in inputs:

            if yield_target == 'source':
                y is None or np.allclose(img,y) or warnings.warn("ignoring 'target' image from input generator")
                target = img
            else:
                target = y

            img.shape == target.shape or _raise(ValueError())

            axes = axes_check_and_normalize(axes)
            _normalize_data = _make_normalize_data(axes)
            # print(axes, img.shape)

            x = img.astype(np.float32, copy=False)

            if psf is not None:
                from scipy.signal import fftconvolve
                # print("blurring with psf")
                _psf = psf.astype(np.float32,copy=False)
                np.min(_psf) >= 0 or _raise(ValueError('psf has negative values.'))
                _psf /= np.sum(_psf)
                if psf_axes is not None:
                    _psf = move_image_axes(_psf, psf_axes, axes, True)
                x.ndim == _psf.ndim or _raise(ValueError('image and psf must have the same number of dimensions.'))

                if 'C' in axes:
                    ch = axes_dict(axes)['C']
                    n_channels = x.shape[ch]
                    # convolve with psf separately for every channel
                    if _psf.shape[ch] == 1:
                        warnings.warn('applying same psf to every channel of the image.')
                    if _psf.shape[ch] in (1,n_channels):
                        x = np.stack([
                            fftconvolve(
                                np.take(x,   i,axis=ch),
                                np.take(_psf,i,axis=ch,mode='clip'),
                                mode='same'
                            )
                            for i in range(n_channels)
                        ],axis=ch)
                    else:
                        raise ValueError('number of psf channels (%d) incompatible with number of image channels (%d).' % (_psf.shape[ch],n_channels))
                else:
                    x = fftconvolve(x, _psf, mode='same')

            if bool(poisson_noise):
                # print("apply poisson noise")
                x = np.random.poisson(np.maximum(0,x).astype(np.int)).astype(np.float32)

            if gauss_sigma > 0:
                # print("adding gaussian noise with sigma = ", gauss_sigma)
                noise = np.random.normal(0,gauss_sigma,size=x.shape).astype(np.float32)
                x = np.maximum(0,x+noise)

            if _subsample != 1:
                # print("down and upsampling X by factor %s" % str(_subsample))
                target = _normalize_data(target)
                x      = _normalize_data(x)

                subsample, subsample_size = _adjust_subsample(x.shape[0],_subsample,crop_threshold)
                # print(subsample, subsample_size)
                if _subsample != subsample:
                    warnings.warn('changing subsample from %s to %s' % (str(_subsample),str(subsample)))

                target = _make_divisible_by_subsample(target,subsample_size)
                x      = _make_divisible_by_subsample(x,     subsample_size)
                x      = _scale_down_up(x,subsample)

                assert x.shape == target.shape, (x.shape, target.shape)

                target = _normalize_data(target,undo=True)
                x      = _normalize_data(x,     undo=True)

            yield x, target, axes, mask


    return Transform('Anisotropic distortion (along %s axis)' % subsample_axis, _generator, 1)



def permute_axes(axes):
    """Transformation to permute images axes.

    Parameters
    ----------
    axes : str
        Target axes, to which the input images will be permuted.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object whose `generator` will
        perform the axes permutation of `x`, `y`, and `mask`.

    """
    axes = axes_check_and_normalize(axes)
    def _generator(inputs):
        for x, y, axes_in, mask in inputs:
            axes_in = axes_check_and_normalize(axes_in)
            if axes_in != axes:
                # print('permuting axes from %s to %s' % (axes_in,axes))
                x = move_image_axes(x, axes_in, axes, True)
                y = move_image_axes(y, axes_in, axes, True)
                if mask is not None:
                    mask = move_image_axes(mask, axes_in, axes)
            yield x, y, axes, mask

    return Transform('Permute axes to %s' % axes, _generator, 1)



def crop_images(slices):
    """Transformation to crop all images (and mask).

    Note that slices must be compatible with the image size.

    Parameters
    ----------
    slices : list or tuple of slice
        List of slices to apply to each dimension of the image.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object whose `generator` will
        perform image cropping of `x`, `y`, and `mask`.

    """
    slices = tuple(slices)
    def _generator(inputs):
        for x, y, axes, mask in inputs:
            axes = axes_check_and_normalize(axes)
            len(axes) == len(slices) or _raise(ValueError())
            yield x[slices], y[slices], axes, (mask[slices] if mask is not None else None)

    return Transform('Crop images (%s)' % str(slices), _generator, 1)



def broadcast_target(target_axes=None):
    """Transformation to broadcast the target image to the shape of the source image.

    Parameters
    ----------
    target_axes : str
        Axes of the target image before broadcasting.
        If `None`, assumed to be the same as for the source image.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object whose `generator` will
        perform broadcasting of `y` to match the shape of `x`.

    """
    def _generator(inputs):
        for x, y, axes_x, mask in inputs:
            if target_axes is not None:
                axes_y = axes_check_and_normalize(target_axes,length=y.ndim)
                y = move_image_axes(y, axes_y, axes_x, True)
            yield x, np.broadcast_to(y,x.shape), axes_x, mask

    return Transform('Broadcast target image to the shape of source', _generator, 1)
