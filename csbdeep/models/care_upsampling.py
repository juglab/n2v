from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from scipy.ndimage.interpolation import zoom

from .care_standard import CARE
from ..data import PercentileNormalizer, PadAndCropResizer
from ..utils import _raise, axes_dict


class UpsamplingCARE(CARE):
    """CARE network for combined image restoration and upsampling of one dimension.

    Extends :class:`csbdeep.models.CARE` by replacing prediction
    (:func:`predict`, :func:`predict_probabilistic`) to first upsample Z before image restoration.
    """

    def predict(self, img, axes, factor, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image with low-resolution Z axis.

        See :func:`CARE.predict` for documentation.

        Parameters
        ----------
        factor : float
            Upsampling factor for Z axis. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.

        """
        img = self._upsample(img, axes, factor)
        return super(UpsamplingCARE, self).predict(img, axes, normalizer, resizer, n_tiles)


    def predict_probabilistic(self, img, axes, factor, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image with low-resolution Z axis for probabilistic prediction.

        See :func:`CARE.predict_probabilistic` for documentation.

        Parameters
        ----------
        factor : float
            Upsampling factor for Z axis. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.

        """
        img = self._upsample(img, axes, factor)
        return super(UpsamplingCARE, self).predict_probabilistic(img, axes, normalizer, resizer, n_tiles)


    @staticmethod
    def _upsample(img, axes, factor, axis='Z'):
        factors = np.ones(img.ndim)
        factors[axes_dict(axes)[axis]] = factor
        return zoom(img,factors,order=1)
