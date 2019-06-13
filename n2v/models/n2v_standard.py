from csbdeep.models import CARE
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict
from csbdeep.internals import nets
from csbdeep.internals.predict import Progress

from six import string_types
from csbdeep.utils.six import Path, FileNotFoundError
from csbdeep.data import PadAndCropResizer

from keras.callbacks import TerminateOnNaN
import tensorflow as tf
from keras import backend as K
from os.path import join

import datetime
import warnings

from .n2v_config import N2VConfig
from ..utils import n2v_utils
from ..internals.N2V_DataWrapper import N2V_DataWrapper
from ..internals.n2v_losses import loss_mae, loss_mse
from ..utils.n2v_utils import pm_identity, pm_normal_additive, pm_normal_fitted, pm_normal_withoutCP, pm_uniform_withCP

import numpy as np

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
        """See class docstring"""
        config is None or isinstance(config, N2VConfig) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))
        (not (config is None and basedir is None)) or _raise(ValueError())

        name is None or isinstance(name, string_types) or _raise(ValueError())
        basedir is None or isinstance(basedir, (string_types, Path)) or _raise(ValueError())
        self.config = config
        self.name = name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.basedir = Path(basedir) if basedir is not None else None
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()
        else:
            config.probabilistic = False


    def _build(self):
        return self._build_unet(
            n_dim           = self.config.n_dim,
            n_channel_out   = self.config.n_channel_out,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
            batch_norm      = self.config.batch_norm
        )(self.config.unet_input_shape)


    def _build_unet(self, n_dim=2, n_depth=2, kern_size=3, n_first=32, n_channel_out=1, residual=True, last_activation='linear', batch_norm=True):
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

        def _build_this(input_shape):
            return nets.custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,) * n_dim,
                               pool_size=(2,) * n_dim, n_channel_out=n_channel_out, residual=residual,
                               prob_out=False, batch_norm=batch_norm)

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
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))
        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)
        div_by = 2**self.config.unet_n_depth
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
                    " (axis %s has incompatible size %d)" % (div_by,axes_relevant,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        manipulator = eval('pm_{0}({1})'.format(self.config.n2v_manipulator, str(self.config.n2v_neighborhood_radius)))

        mean, std = float(self.config.mean), float(self.config.std)

        X = self.__normalize__(X, mean, std)
        validation_X = self.__normalize__(validation_X, mean, std)

        # Here we prepare the Noise2Void data. Our input is the noisy data X and as target we take X concatenated with
        # a masking channel. The N2V_DataWrapper will take care of the pixel masking and manipulating.
        training_data = N2V_DataWrapper(X, np.concatenate((X, np.zeros(X.shape, dtype=X.dtype)), axis=axes.index('C')),
                                                    self.config.train_batch_size, int(train_num_pix/100 * self.config.n2v_perc_pix),
                                                    self.config.n2v_patch_shape, manipulator)

        # validation_Y is also validation_X plus a concatinated masking channel.
        # To speed things up, we precomupte the masking vo the validation data.
        validation_Y = np.concatenate((validation_X, np.zeros(validation_X.shape, dtype=validation_X.dtype)), axis=axes.index('C'))
        n2v_utils.manipulate_val_data(validation_X, validation_Y,
                                                        num_pix=int(val_num_pix/100 * self.config.n2v_perc_pix),
                                                        shape=val_patch_shape,
                                                        value_manipulation=manipulator)

        history = self.keras_model.fit_generator(generator=training_data, validation_data=(validation_X, validation_Y),
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
            from keras.optimizers import Adam
            optimizer = Adam(lr=self.config.train_learning_rate)
        self.callbacks = self.prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.basedir is not None:
            if self.config.train_checkpoint is not None:
                from keras.callbacks import ModelCheckpoint
                self.callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True,  save_weights_only=True))
                self.callbacks.append(ModelCheckpoint(str(self.logdir / 'weights_now.h5'),             save_best_only=False, save_weights_only=True))

            if self.config.train_tensorboard:
                from csbdeep.utils.tf import CARETensorBoard

                class N2VTensorBoard(CARETensorBoard):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}

                        if self.validation_data and self.freq:
                            if epoch % self.freq == 0:
                                # TODO: implement batched calls to sess.run
                                # (current call will likely go OOM on GPU)
                                if self.model.uses_learning_phase:
                                    cut_v_data = len(self.model.inputs)
                                    val_data = [self.validation_data[0][:self.n_images]] + [0]
                                    tensors = self.model.inputs + [K.learning_phase()]
                                else:
                                    val_data = list(v[:self.n_images] for v in self.validation_data)
                                    tensors = self.model.inputs
                                feed_dict = dict(zip(tensors, val_data))
                                result = self.sess.run([self.merged], feed_dict=feed_dict)
                                summary_str = result[0]

                                self.writer.add_summary(summary_str, epoch)

                        for name, value in logs.items():
                            if name in ['batch', 'size']:
                                continue
                            summary = tf.Summary()
                            summary_value = summary.value.add()
                            summary_value.simple_value = value.item()
                            summary_value.tag = name
                            self.writer.add_summary(summary, epoch)

                        self.writer.flush()

                self.callbacks.append(N2VTensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=False))

        if self.config.train_reduce_lr is not None:
            from keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def prepare_model(self, model, optimizer, loss, metrics=('mse', 'mae')):
        """ TODO """

        from keras.optimizers import Optimizer
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


    def __normalize__(self, data, mean, std):
        return (data - mean)/std


    def __denormalize__(self, data, mean, std):
        return (data * std) + mean


    def predict(self, img, axes, resizer=PadAndCropResizer(), n_tiles=None):
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

        Returns
        -------
        image : array(float)
                The restored image.
        """
        mean, std = float(self.config.mean), float(self.config.std)

        normalized = self.__normalize__(img, mean, std)

        pred = self._predict_mean_and_scale(normalized, axes=axes, normalizer=None, resizer=resizer, n_tiles=n_tiles)[0]

        return self.__denormalize__(pred, mean, std)
