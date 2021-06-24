import argparse

import tensorflow.keras.backend as K

from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict, backend_channels_last

from six import string_types

import numpy as np

# This class is a adapted version of csbdeep.models.config.py.
class N2VConfig(argparse.Namespace):
    """Default configuration for a N2V trainable CARE model.

    This class is meant to be used with :class:`N2V`.

    Parameters
    ----------
    X      : array(float)
             The training data 'X', with dimensions 'SZYXC' or 'SYXC'
    kwargs : dict
             Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> n2v_config = N2VConfig(X, unet_n_depth=3)

    Attributes
    ----------
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    batch_norm : bool
        Activate batch norm
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
    n2v_perc_pix : float
        Percentage of pixel to manipulate per patch. Default: ``1.5``
    n2v_patch_shape : tuple
        Random patches of this shape are extracted from the given training data. Default: ``(64, 64) if n_dim==2 else (64, 64, 64)``
    n2v_manipulator : str
        Noise2Void pixel value manipulator. Default: ``uniform_withCP``
    n2v_neighborhood_radius : int
        Neighborhood radius for n2v manipulator. Default: ``5``
    single_net_per_channel : bool
        Enabling this creates a unet for each channel and each channel will be treated independently.
        Note: This makes the ``network n_channel_in`` times larger. Default: ``True``
    structN2Vmask : [[int]]
        Masking kernel for StructN2V to hide pixels adjacent to main blind spot. Value 1 = 'hidden', Value 0 = 'non hidden'. Nested lists equivalent to ndarray. Must have odd length in each dimension (center pixel is blind spot). Default ``None`` implies normal N2V masking.


        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, X, **kwargs):
        
        # X is empty if config is None
        if (X.size != 0):
    
            assert len(X.shape) == 4 or len(X.shape) == 5, "Only 'SZYXC' or 'SYXC' as dimensions is supported."
    
            n_dim = len(X.shape) - 2
            n_channel_in = X.shape[-1]
            n_channel_out = n_channel_in

            means, stds = [], []
            for i in range(n_channel_in):
                means.append(np.mean(X[...,i]))
                stds.append(np.std(X[...,i]))

            if n_dim == 2:
                axes = 'SYXC'
            elif n_dim == 3:
                axes = 'SZYXC'
    
            # parse and check axes
            axes = axes_check_and_normalize(axes)
            ax = axes_dict(axes)
            ax = {a: (ax[a] is not None) for a in ax}
    
            (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
            not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))
    
            axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
            axes = axes.replace('S','') # remove sample axis if it exists
    
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
    
            # normalization parameters
            self.means                 = [str(el) for el in means]
            self.stds                  = [str(el) for el in stds]
            # directly set by parameters
            self.n_dim                 = n_dim
            self.axes                  = axes
            self.n_channel_in          = int(n_channel_in)
            self.n_channel_out         = int(n_channel_out)
    
            # default config (can be overwritten by kwargs below)
            self.unet_residual         = False
            self.unet_n_depth          = 2
            self.unet_kern_size        = 5 if self.n_dim==2 else 3
            self.unet_n_first          = 32
            self.unet_last_activation  = 'linear'
            if backend_channels_last():
                self.unet_input_shape  = self.n_dim*(None,) + (self.n_channel_in,)
            else:
                self.unet_input_shape  = (self.n_channel_in,) + self.n_dim*(None,)
    
            self.train_loss            = 'mae'
            self.train_epochs          = 100
            self.train_steps_per_epoch = 400
            self.train_learning_rate   = 0.0004
            self.train_batch_size      = 16
            self.train_tensorboard     = True
            self.train_checkpoint      = 'weights_best.h5'
            self.train_reduce_lr       = {'factor': 0.5, 'patience': 10}
            self.batch_norm            = True
            self.n2v_perc_pix           = 1.5
            self.n2v_patch_shape       = (64, 64) if self.n_dim==2 else (64, 64, 64)
            self.n2v_manipulator       = 'uniform_withCP'
            self.n2v_neighborhood_radius = 5

            self.single_net_per_channel = True

            # disallow setting 'n_dim' manually
            try:
                del kwargs['n_dim']
                # warnings.warn("ignoring parameter 'n_dim'")
            except:
                pass

            self.structN2Vmask = None
            
        self.probabilistic         = False

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
        ok['means'] = True
        for mean in self.means:
            ok['means'] &= np.isscalar(float(mean))
        ok['stds'] = True
        for std in self.stds:
            ok['stds'] &= np.isscalar(float(std))  and float(std) > 0.0
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except:
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)

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
            (self.train_loss in ('mse','mae'))
        )
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)
        ok['batch_norm']            = isinstance(self.batch_norm, bool)
        ok['n2v_perc_pix']           = self.n2v_perc_pix > 0 and self.n2v_perc_pix <= 100
        ok['n2v_patch_shape']       = (
            isinstance(self.n2v_patch_shape, (list,tuple)) and
            len(self.n2v_patch_shape) == self.n_dim and
            all(d > 0 for d in self.n2v_patch_shape)
        )
        ok['n2v_manipulator']       = self.n2v_manipulator in ['normal_withoutCP', 'uniform_withCP', 'normal_additive',
                                                               'normal_fitted', 'identity']
        ok['n2v_neighborhood_radius']= _is_int(self.n2v_neighborhood_radius, 0) 
        ok['single_net_per_channel'] = isinstance( self.single_net_per_channel, bool )

        if self.structN2Vmask is None:
            ok['structN2Vmask'] = True
        else:
            mask = np.array(self.structN2Vmask)
            t1 = mask.ndim == self.n_dim
            t2 = all(x%2==1 for x in mask.shape)
            t3 = all([x in [0,1] for x in mask.flat])
            ok['structN2Vmask'] = t1 and t2 and t3

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())

    def update_parameters(self, allow_new=True, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])
