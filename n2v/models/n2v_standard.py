import datetime
import warnings
from datetime import datetime
from typing import Tuple, List, Dict, Union

import numpy as np
import tensorflow as tf
from csbdeep.data import PadAndCropResizer
from csbdeep.models import CARE
from csbdeep.models.base_model import suppress_without_basedir
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict, load_json, save_json
from csbdeep.utils.six import Path, FileNotFoundError
from csbdeep.utils.tf import CARETensorBoardImage
from six import string_types
from tensorflow.keras.callbacks import TerminateOnNaN

from .n2v_config import N2VConfig
from ..internals.N2V_DataWrapper import N2V_DataWrapper
from ..internals.n2v_losses import loss_mse, loss_mae
from ..utils import n2v_utils
from ..utils.n2v_utils import pm_identity, pm_normal_additive, \
    pm_normal_fitted, pm_normal_withoutCP, pm_uniform_withCP, \
    pm_uniform_withoutCP, pm_mean, pm_median, \
    tta_forward, tta_backward
from ..nets.unet import build_single_unet_per_channel, build_unet
from ..utils.export_utils import which_algorithm, generate_bioimage_md, build_modelzoo, \
    Extensions, save_model_tf, get_algorithm_details


class N2V(CARE):
    """The Noise2Void training scheme to train a standard CARE network for image restoration and enhancement.

        Uses a convolutional neural network created by :func:`csbdeep.internals.nets.custom_unet`.

        Parameters
        ----------
        config : :class:`n2v.models.N2VConfig` or None
            Valid configuration of N2V network (see :func:`N2VConfig.is_valid`).
            Will be saved to disk as JSON (``config.json``).
            If set to ``None``, will be loaded from disk (must exist).
        name : str or None
            Model name. Uses a timestamp if set to ``None`` (default).
        basedir : str
            Directory that contains (or will contain) a folder with the given model name.
            Use ``None`` to disable saving (or loading) any data to (or from) disk (regardless of other parameters).

        Raises
        ------
        FileNotFoundError
            If ``config=None`` and config cannot be loaded from disk.
        ValueError
            Illegal arguments, including invalid configuration.

        Example
        -------
        >>> model = N2V(config, 'my_model')

        Attributes
        ----------
        config : :class:`n2v.models.N2VConfig`
            Configuration of N2V trainable CARE network, as provided during instantiation.
        keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
            Keras neural network model.
        name : str
            Model name.
        logdir : :class:`pathlib.Path`
            Path to model folder (which stores configuration, weights, etc.)
        """

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""

        config is None or isinstance(config, self._config_class) or _raise(
            ValueError("Invalid configuration of type '%s', was expecting type '%s'." % (
                type(config).__name__, self._config_class.__name__))
        )
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))
        (not (config is None and basedir is None)) or _raise(
            ValueError("No config provided and cannot be loaded from disk since basedir=None."))

        name is None or (isinstance(name, string_types) and len(name) > 0) or _raise(
            ValueError("No valid name: '%s'" % str(name)))
        basedir is None or isinstance(basedir, (string_types, Path)) or _raise(
            ValueError("No valid basedir: '%s'" % str(basedir)))
        self.config = config
        self.name = name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.basedir = Path(basedir) if basedir is not None else None
        if config is not None:
            # config was provided -> update before it is saved to disk
            self._update_and_check_config()
        self._set_logdir()
        if config is None:
            # config was loaded from disk -> update it after loading
            self._update_and_check_config()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()

    def _build(self):
        return self._build_unet(
            n_dim=self.config.n_dim,
            residual=self.config.unet_residual,
            n_depth=self.config.unet_n_depth,
            kern_size=self.config.unet_kern_size,
            n_first=self.config.unet_n_first,
            last_activation=self.config.unet_last_activation,
            batch_norm=self.config.batch_norm,
            blurpool=self.config.blurpool,
            skip_skipone=self.config.skip_skipone
        )(self.config.unet_input_shape, self.config.single_net_per_channel)

    def _build_unet(self, n_dim=2, n_depth=2, kern_size=3, n_first=32, residual=True,
                    last_activation='linear', batch_norm=True,
                    blurpool=False,
                    skip_skipone=False):
        """Construct a common CARE neural net based on U-Net [1]_ and residual learning [2]_ to be used for image restoration/enhancement.
           Parameters
           ----------
           n_dim : int
               number of image dimensions (2 or 3)
           n_depth : int
               number of resolution levels of U-Net architecture
           kern_size : int
               size of convolution filter in all image dimensions
           n_first : int
               number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
           n_channel_out : int
               number of channels of the predicted output image
           residual : bool
               if True, model will internally predict the residual w.r.t. the input (typically better)
               requires number of input and output image channels to be equal
           last_activation : str
               name of activation function for the final output layer
           batch_norm : bool
               Use batch normalization during training
           Returns
           -------
           function
               Function to construct the network, which takes as argument the shape of the input image
           Example
           -------
           >>> model = common_unet(2, 2, 3, 32, 1, True, 'linear', True)(input_shape)
           References
           ----------
           .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
           .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
           """

        def _build_this(input_shape, single_net_per_channel):
            if single_net_per_channel:
                return build_single_unet_per_channel(input_shape, last_activation, n_depth, n_first,
                                                     (kern_size,) * n_dim,
                                                     pool_size=(2,) * n_dim, residual=residual, prob_out=False,
                                                     batch_norm=batch_norm,
                                                     blurpool=blurpool,
                                                     skip_skipone=skip_skipone)
            else:
                return build_unet(input_shape,
                                  last_activation,
                                  n_depth,
                                  n_first,
                                  (kern_size,) * n_dim,
                                  pool_size=(2,) * n_dim,
                                  residual=residual,
                                  prob_out=False,
                                  batch_norm=batch_norm,
                                  blurpool=blurpool,
                                  skip_skipone=skip_skipone)

        return _build_this

    def train(self, X, validation_X, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        validation_x : :class:`numpy.ndarray`
            Array of validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """

        n_train, n_val = len(X), len(validation_X)
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100 * frac_val))
        axes = axes_check_and_normalize('S' + self.config.axes, X.ndim)
        ax = axes_dict(axes)
        div_by = 2 ** self.config.unet_n_depth
        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
        val_num_pix = 1
        train_num_pix = 1
        val_patch_shape = ()
        for a in axes_relevant:
            n = X.shape[ax[a]]
            val_num_pix *= validation_X.shape[ax[a]]
            train_num_pix *= X.shape[ax[a]]
            val_patch_shape += tuple([validation_X.shape[ax[a]]])
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axes %s"
                    " (axis %s has incompatible size %d)" % (div_by, axes_relevant, a, n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        manipulator = eval('pm_{0}({1})'.format(self.config.n2v_manipulator, str(self.config.n2v_neighborhood_radius)))

        means = np.array([float(mean) for mean in self.config.means], ndmin=len(X.shape), dtype=np.float32)
        stds = np.array([float(std) for std in self.config.stds], ndmin=len(X.shape), dtype=np.float32)

        X = self.__normalize__(X, means, stds)
        validation_X = self.__normalize__(validation_X, means, stds)

        # Here we prepare the Noise2Void data. Our input is the noisy data X and as target we take X concatenated with
        # a masking channel. The N2V_DataWrapper will take care of the pixel masking and manipulating.
        _mask = np.array(self.config.structN2Vmask) if self.config.structN2Vmask else None
        training_data = N2V_DataWrapper(X, np.concatenate((X, np.zeros(X.shape, dtype=X.dtype)), axis=axes.index('C')),
                                        batch_size=self.config.train_batch_size,
                                        length=self.config.train_steps_per_epoch * self.config.train_epochs,
                                        perc_pix=self.config.n2v_perc_pix,
                                        shape=self.config.n2v_patch_shape,
                                        value_manipulation=manipulator, structN2Vmask=_mask)

        # validation_Y is also validation_X plus a concatenated masking channel.
        # To speed things up, we precompute the masking for the validation data.
        validation_Y = np.concatenate((validation_X, np.zeros(validation_X.shape, dtype=validation_X.dtype)),
                                      axis=axes.index('C'))
        n2v_utils.manipulate_val_data(validation_X, validation_Y,
                                      perc_pix=self.config.n2v_perc_pix,
                                      shape=val_patch_shape,
                                      value_manipulation=manipulator)
        self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=(validation_X, validation_X),
                                                   log_dir=str(self.logdir / 'logs' / 'images'),
                                                   n_images=3, prob_out=False))

        history = self.keras_model.fit(iter(training_data), validation_data=(validation_X, validation_Y),
                                       epochs=epochs, steps_per_epoch=steps_per_epoch,
                                       callbacks=self.callbacks, verbose=1)

        if self.basedir is not None:
            self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

            if self.config.train_checkpoint is not None:
                print()
                self._find_and_load_weights(self.config.train_checkpoint)
                try:
                    # remove temporary weights
                    (self.logdir / 'weights_now.h5').unlink()
                except FileNotFoundError:
                    pass

        return history

    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.

        Calls :func:`csbdeep.internals.train.prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`csbdeep.internals.train.prepare_model`.

        """
        if optimizer is None:
            from tensorflow.keras.optimizers import Adam
            optimizer = Adam(learning_rate=self.config.train_learning_rate)
        self.callbacks = self.prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.basedir is not None:
            if self.config.train_checkpoint is not None:
                from tensorflow.keras.callbacks import ModelCheckpoint
                self.callbacks.append(
                    ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True,
                                    save_weights_only=True))
                self.callbacks.append(
                    ModelCheckpoint(str(self.logdir / 'weights_now.h5'), save_best_only=False, save_weights_only=True))

            if self.config.train_tensorboard:
                from tensorflow.keras.callbacks import TensorBoard
                self.callbacks.append(
                    TensorBoard(log_dir=str(self.logdir / 'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True

    def prepare_model(self, model, optimizer, loss, metrics=('mse', 'mae')):
        """ TODO """

        from tensorflow.keras.optimizers import Optimizer
        isinstance(optimizer, Optimizer) or _raise(ValueError())

        if loss == 'mse':
            loss_standard = eval('loss_mse()')
        elif loss == 'mae':
            loss_standard = eval('loss_mae()')

        _metrics = [eval('loss_%s()' % m) for m in metrics]
        callbacks = [TerminateOnNaN()]

        # compile model
        model.compile(optimizer=optimizer, loss=loss_standard, metrics=_metrics)

        return callbacks

    def __normalize__(self, data, means, stds):
        return (data - means) / stds

    def __denormalize__(self, data, means, stds):
        return (data * stds) + means

    def predict(self, img, axes, resizer=PadAndCropResizer(), n_tiles=None, tta=False):
        """
        Apply the network to sofar unseen data. This method expects the raw data, i.e. not normalized.
        During prediction the mean and standard deviation, stored with the model (during data generation), are used
        for normalization.

        Parameters
        ----------
        img     : array(floats)
                  The raw images.
        axes    : String
                  Axes of the image ('YX').
        resizer : class(Resizer), optional(default=PadAndCropResizer())
        n_tiles : tuple(int)
                  Number of tiles to tile the image into, if it is too large for memory.
        tta     : bool
                  Use test-time augmentation during prediction.

        Returns
        -------
        image : array(float)
                The restored image.
        """
        means = np.array([float(mean) for mean in self.config.means], ndmin=len(img.shape), dtype=np.float32)
        stds = np.array([float(std) for std in self.config.stds], ndmin=len(img.shape), dtype=np.float32)

        if img.dtype != np.float32:
            print('The input image is of type {} and will be casted to float32 for prediction.'.format(img.dtype))
            img = img.astype(np.float32)

        new_axes = axes
        new_n_tiles = n_tiles
        if 'C' in axes:
            new_axes = axes.replace('C', '') + 'C'
            if n_tiles:
                new_n_tiles = tuple([n_tiles[axes.index(c)] for c in axes if c != 'C']) + (n_tiles[axes.index('C')],)
            normalized = self.__normalize__(np.moveaxis(img, axes.index('C'), -1), means, stds)
        else:
            normalized = self.__normalize__(img[..., np.newaxis], means, stds)
            normalized = normalized[..., 0]

        if tta:
            aug = tta_forward(normalized)
            preds = []
            for img in aug:
                preds.append(self._predict_mean_and_scale(img, axes=new_axes, normalizer=None, resizer=resizer,
                                             n_tiles=new_n_tiles)[0])
            pred = tta_backward(preds)
        else:
            pred = \
                self._predict_mean_and_scale(normalized, axes=new_axes, normalizer=None, resizer=resizer,
                                             n_tiles=new_n_tiles)[0]

        pred = self.__denormalize__(pred, means, stds)

        if 'C' in axes:
            pred = np.moveaxis(pred, -1, axes.index('C'))

        return pred

    # same as predict, but input and output are converted to float64 to comply with bioimage core
    def predict_bioimageio(self, img: np.ndarray, axes: str, eps: float = 1e-6):
        means = np.array([float(mean) for mean in self.config.means], ndmin=len(img.shape))
        stds = np.array([float(std) for std in self.config.stds], ndmin=len(img.shape))

        img = img.astype(np.float64)
        if 'b' in axes:
            axes = axes.replace('b', 'S').upper()
        new_axes = axes
        if 'C' in axes:
            new_axes = axes.replace('C', '') + 'C'
            normalized = np.moveaxis(img, axes.index('C'), -1)
            normalized = (normalized - means) / (stds + eps)
        else:
            normalized = img[..., np.newaxis]
            normalized = (normalized - means) / (stds + eps)
            normalized = normalized[..., 0]

        pred = self._predict_mean_and_scale(normalized, axes=new_axes, normalizer=None, resizer=None)[0]
        pred = pred.astype(np.float64)
        pred = self.__denormalize__(pred, means, stds)
        if 'C' in axes:
            pred = np.moveaxis(pred, -1, axes.index('C'))

        return pred

    def _set_logdir(self):
        self.logdir = self.basedir / self.name

        config_file = self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                self.config = self._config_class(np.array([]), **config_dict)
                if not self.config.is_valid():
                    invalid_attr = self.config.is_valid(True)[1]
                    raise ValueError('Invalid attributes in loaded config: ' + ', '.join(invalid_attr))
            else:
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
        else:
            if self.logdir.exists():
                warnings.warn(
                    'output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), str(config_file))

    @suppress_without_basedir(warn=True)
    def export_TF(
            self,
            name: str,
            description: str,
            authors: List[str],
            test_img: np.ndarray,
            axes: str,
            patch_shape: Tuple[int, int],
            license: str = 'BSD-3-Clause',
            result_path: Union[Path, str] = None
    ):
        """
        name: String
            Name of the model. 
        description: String
            A short description of the model e.g. on what data it was trained.
        authors: List
            Comma seperated list of author names.
        patch_shape: Tuple
            The shape of the patches used in model.train().
        licence: String
            Model license, default is BSD-3-Clause
        result_path: String
            Path to the result folder, optional
        """
        input_n_dims = len(test_img.shape)
        if 'C' in axes:
            input_n_dims -= 1
        assert input_n_dims == self.config.n_dim, \
            'Input and network dimensions do not match.'
        assert test_img.shape[axes.index('X')] == test_img.shape[axes.index('Y')], \
            'X and Y dimensions are not of same length.'

        if patch_shape != None:
            self.config.patch_shape = patch_shape

        if result_path is None:
            result_path = self.logdir
        result_path = Path(result_path).absolute()

        test_output = self.predict_bioimageio(test_img, axes)

        model_path = result_path / 'tf_model.zip'
        config_path = result_path / 'config.json'
        save_model_tf(
            model=self.keras_model,
            config=self.config,
            config_path=config_path,
            model_path=model_path
        )

        new_axes = axes.replace('S', 'b').lower()
        if 'b' not in new_axes:
            new_axes = 'b' + new_axes
            axes = 'S' + axes
            test_img = test_img[np.newaxis, ...]
            test_output = test_output[np.newaxis, ...]

        input_file = self.logdir.absolute() / 'test_input.npy'
        np.save(input_file, test_img.astype(np.float64))

        output_file = self.logdir.absolute() / 'test_output.npy'
        np.save(output_file, test_output.astype(np.float64))

        preprocessing = [{
            'name': 'zero_mean_unit_variance',
            'kwargs': {
                'mode': 'fixed',
                'axes': 'yx' if len(axes) == 4 else 'zyx',
                'mean': [float(m) for m in self.config.means],
                'std': [float(s) for s in self.config.stds]
            }
        }]
        postprocessing = [{
            'name': 'scale_linear',
            'kwargs': {
                'axes': 'yx' if len(axes) == 4 else 'zyx',
                'gain': [float(s) for s in self.config.stds],
                'offset': [float(m) for m in self.config.means]
            }
        }]
        authors = [{"name": author} for author in authors]

        algorithm = which_algorithm(self.config)
        cite = get_algorithm_details(algorithm)
        doc = generate_bioimage_md(name, cite, result_path)
        files = [str(config_path.absolute()), str(model_path.absolute())]
        result_archive_path = result_path / (result_path.stem + Extensions.BIOIMAGE_EXT.value)

        build_modelzoo(
            result_archive_path,
            model_path,
            result_path,
            input_file,
            output_file,
            preprocessing,
            postprocessing,
            doc,
            name,
            authors,
            algorithm,
            tf.__version__,
            cite,
            new_axes,
            files
        )

        print("\nModel exported in BioImage ModelZoo format:\n%s" % str(result_archive_path.resolve()))

    @property
    def _config_class(self):
        return N2VConfig
