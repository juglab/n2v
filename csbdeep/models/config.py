from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np
import argparse
import warnings

import keras.backend as K

from ..utils import _raise, axes_check_and_normalize, axes_dict, backend_channels_last


class Config(argparse.Namespace):
    """Default configuration for a CARE model.

    This configuration is meant to be used with :class:`CARE`
    and related models (e.g., :class:`IsotropicCARE`).

    Parameters
    ----------
    axes : str
        Axes of the neural network (channel axis optional).
    n_channel_in : int
        Number of channels of given input image.
    n_channel_out : int
        Number of channels of predicted output image.
    probabilistic : bool
        Probabilistic prediction of per-pixel Laplace distributions or
        typical regression of per-pixel scalar values.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> config = Config('YX', probabilistic=True, unet_n_depth=3)

    Attributes
    ----------
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: ``'laplace' if probabilistic else 'mae'``
    train_epochs : int
        Number of training epochs. Default: ``100``
    train_steps_per_epoch : int
        Number of parameter update steps per epoch. Default: ``400``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_batch_size : int
        Batch size for training. Default: ``16``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable. Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable. Default: ``{'factor': 0.5, 'patience': 10}``
    batch_norm : bool
        Activate batch norm
    train_scheme : str
        Training scheme to switch between ``CARE`` and ``Noise2Void``. Default: ``CARE``
    n2v_num_pix : int
        Number of pixel to manipulate per patch. Default: 1
    n2v_patch_shape : tuple
        Random patches of this shape are extracted from the given training data. Default: (64, 64) or (64, 64, 64)
    n2v_manipulator : str
        Noise2Void pixel value manipulator. Default: ``uniform_withCP``
    n2v_neighborhood_radius : str
        Neighborhood radius for n2v manipulator. Default: ``5``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes, n_channel_in=1, n_channel_out=1, probabilistic=False, **kwargs):
        """See class docstring."""

        # parse and check axes
        axes = axes_check_and_normalize(axes)
        ax = axes_dict(axes)
        ax = {a: (ax[a] is not None) for a in ax}

        (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
        not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))

        axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
        axes = axes.replace('S','') # remove sample axis if it exists

        n_dim = 3 if (ax['Z'] or ax['T']) else 2

        # TODO: Config not independent of backend. Problem?
        # could move things around during train/predict as an alternative... good idea?
        # otherwise, users can choose axes of input image anyhow, so doesn't matter if model is fixed to something else
        if backend_channels_last():
            if ax['C']:
                axes[-1] == 'C' or _raise(ValueError('channel axis must be last for backend (%s).' % K.backend()))
            else:
                axes += 'C'
        else:
            if ax['C']:
                axes[0] == 'C' or _raise(ValueError('channel axis must be first for backend (%s).' % K.backend()))
            else:
                axes = 'C'+axes

        # directly set by parameters
        self.n_dim                 = n_dim
        self.axes                  = axes
        self.n_channel_in          = int(n_channel_in)
        self.n_channel_out         = int(n_channel_out)
        self.probabilistic         = bool(probabilistic)

        # default config (can be overwritten by kwargs below)
        self.unet_residual         = self.n_channel_in == self.n_channel_out
        self.unet_n_depth          = 2
        self.unet_kern_size        = 5 if self.n_dim==2 else 3
        self.unet_n_first          = 32
        self.unet_last_activation  = 'linear'
        if backend_channels_last():
            self.unet_input_shape  = self.n_dim*(None,) + (self.n_channel_in,)
        else:
            self.unet_input_shape  = (self.n_channel_in,) + self.n_dim*(None,)

        self.train_loss            = 'laplace' if self.probabilistic else 'mae'
        self.train_epochs          = 100
        self.train_steps_per_epoch = 400
        self.train_learning_rate   = 0.0004
        self.train_batch_size      = 16
        self.train_tensorboard     = True
        self.train_checkpoint      = 'weights_best.h5'
        self.train_reduce_lr       = {'factor': 0.5, 'patience': 10, 'min_delta': 0}
        self.batch_norm            = False
        self.train_scheme          = 'CARE'
        self.n2v_num_pix           = 1
        self.n2v_patch_shape       = (64, 64) if self.n_dim==2 else (64, 64, 64)
        self.n2v_manipulator       = 'uniform_withCP'
        self.n2v_neighborhood_radius = '5'

        # disallow setting 'n_dim' manually
        try:
            del kwargs['n_dim']
            # warnings.warn("ignoring parameter 'n_dim'")
        except:
            pass

        for k in kwargs:
            setattr(self, k, kwargs[k])


    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.

        Returns
        -------
        bool
            Flag that indicates whether the current configuration values are valid.
        """
        def _is_int(v,low=None,high=None):
            return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
            )

        ok = {}
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except:
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)
        ok['probabilistic'] = isinstance(self.probabilistic,bool)

        ok['unet_residual'] = (
            isinstance(self.unet_residual,bool) and
            (not self.unet_residual or (self.n_channel_in==self.n_channel_out))
        )
        ok['unet_n_depth']         = _is_int(self.unet_n_depth,1)
        ok['unet_kern_size']       = _is_int(self.unet_kern_size,1)
        ok['unet_n_first']         = _is_int(self.unet_n_first,1)
        ok['unet_last_activation'] = self.unet_last_activation in ('linear','relu')
        ok['unet_input_shape'] = (
            isinstance(self.unet_input_shape,(list,tuple)) and
            len(self.unet_input_shape) == self.n_dim+1 and
            self.unet_input_shape[-1] == self.n_channel_in and
            all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1]))
        )
        ok['train_loss'] = (
            (    self.probabilistic and self.train_loss == 'laplace'   ) or
            (not self.probabilistic and self.train_loss in ('mse','mae'))
        )
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)
        ok['batch_norm']            = isinstance(self.batch_norm, bool)
        ok['train_scheme']          = (
            self.train_scheme in ('CARE', 'Noise2Void') and
            ( True if self.train_scheme == 'Noise2Void' and not self.probabilistic else False)
        )
        ok['n2v_num_pix']           = _is_int(self.n2v_num_pix, 1)
        ok['n2v_patch_shape']       = (
            isinstance(self.n2v_patch_shape, (list,tuple)) and
            len(self.n2v_patch_shape) == self.n_dim and
            all(d > 0 for d in self.n2v_patch_shape)
        )
        ok['n2v_manipulator']       = self.n2v_manipulator in ['normal_withoutCP', 'uniform_withCP', 'normal_additive',
                                                               'normal_fitted', 'identity']

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())
