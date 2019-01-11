from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from scipy.ndimage.interpolation import zoom

from csbdeep.internals.probability import ProbabilisticPrediction
from .care_standard import CARE
from ..internals.predict import predict_direct
from ..data import PercentileNormalizer, PadAndCropResizer
from ..utils import _raise, axes_check_and_normalize


class IsotropicCARE(CARE):
    """CARE network for isotropic image reconstruction.

    Extends :class:`csbdeep.models.CARE` by replacing prediction
    (:func:`predict`, :func:`predict_probabilistic`) to do isotropic reconstruction.
    """

    def predict(self, img, axes, factor, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), batch_size=8):
        """Apply neural network to raw image for isotropic reconstruction.

        See :func:`CARE.predict` for documentation.

        Parameters
        ----------
        factor : float
            Upsampling factor for Z axis. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.
        batch_size : int
            Number of image slices that are processed together by the neural network.
            Reduce this value if out of memory errors occur.

        """
        return self._predict_mean_and_scale(img, axes, factor, normalizer, resizer, batch_size)[0]


    def predict_probabilistic(self, img, axes, factor, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), batch_size=8):
        """Apply neural network to raw image to predict probability distribution for isotropic restored image.

        See :func:`CARE.predict_probabilistic` for documentation.

        Parameters
        ----------
        factor : float
            Upsampling factor for Z axis. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.
        batch_size : int
            Number of image slices that are processed together by the neural network.
            Reduce this value if out of memory errors occur.

        """
        self.config.probabilistic or _raise(ValueError('This is not a probabilistic model.'))
        mean, scale = self._predict_mean_and_scale(img, axes, factor, normalizer, resizer, batch_size)
        return ProbabilisticPrediction(mean, scale)


    def _predict_mean_and_scale(self, img, axes, factor, normalizer, resizer, batch_size):
        """Apply neural network to raw image to restore isotropic resolution.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        axes = axes_check_and_normalize(axes,img.ndim)
        'Z' in axes or _raise(ValueError())
        axes_tmp = 'CZ' + axes.replace('Z','').replace('C','')
        _permute_axes = self._make_permute_axes(axes, axes_tmp)
        channel = 0

        x = _permute_axes(img)

        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        np.isscalar(factor) and factor > 0 or _raise(ValueError())

        def scale_z(arr,factor):
            return zoom(arr,(1,factor,1,1),order=1)

        # normalize
        x = normalizer.before(x,axes_tmp)

        # scale z up (second axis)
        x_scaled = scale_z(x,factor)

        # resize: make (x,y,z) image dimensions divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x_scaled = resizer.before(x_scaled,div_n,exclude=channel)

        # move channel to the end
        x_scaled = np.moveaxis(x_scaled, channel, -1)
        channel = -1

        # u1: first rotation and prediction
        x_rot1   = self._rotate(x_scaled, axis=1, copy=False)
        u_rot1   = predict_direct(self.keras_model, x_rot1, channel_in=channel, channel_out=channel, single_sample=False,
                                  batch_size=batch_size, verbose=0)
        u1       = self._rotate(u_rot1, -1, axis=1, copy=False)

        # u2: second rotation and prediction
        x_rot2   = self._rotate(self._rotate(x_scaled, axis=2, copy=False), axis=0, copy=False)
        u_rot2   = predict_direct(self.keras_model, x_rot2, channel_in=channel, channel_out=channel, single_sample=False,
                                  batch_size=batch_size, verbose=0)
        u2       = self._rotate(self._rotate(u_rot2, -1, axis=0, copy=False), -1, axis=2, copy=False)

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        u_rot1.shape[channel] == n_channel_predicted or _raise(ValueError())
        u_rot2.shape[channel] == n_channel_predicted or _raise(ValueError())

        # move channel back to the front
        u1 = np.moveaxis(u1, channel, 0)
        u2 = np.moveaxis(u2, channel, 0)
        channel = 0

        # resize after prediction
        u1 = resizer.after(u1,exclude=channel)
        u2 = resizer.after(u2,exclude=channel)

        # combine u1 & u2
        mean1, scale1 = self._mean_and_scale_from_prediction(u1,axis=channel)
        mean2, scale2 = self._mean_and_scale_from_prediction(u2,axis=channel)
        # avg = lambda u1,u2: (u1+u2)/2 # arithmetic mean
        avg = lambda u1,u2: np.sqrt(np.maximum(u1,0)*np.maximum(u2,0)) # geometric mean
        mean, scale = avg(mean1,mean2), None
        if self.config.probabilistic:
            scale = np.maximum(scale1,scale2)

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            mean, scale = normalizer.after(mean, scale)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)

        return mean, scale


    @staticmethod
    def _rotate(arr, k=1, axis=1, copy=True):
        """Rotate by 90 degrees around the first 2 axes."""
        if copy:
            arr = arr.copy()

        k = k % 4

        arr = np.rollaxis(arr, axis, arr.ndim)

        if k == 0:
            res = arr
        elif k == 1:
            res = arr[::-1].swapaxes(0, 1)
        elif k == 2:
            res = arr[::-1, ::-1]
        else:
            res = arr.swapaxes(0, 1)[::-1]

        res = np.rollaxis(res, -1, axis)
        return res

