Model training
==============

Given suitable training data (see :doc:`datagen`), we can define
and train a CARE model to restore the source data.
To that end, we first need to specify all the options of the model
by creating a configuration object via :class:`csbdeep.models.Config`.
Note that we provide sensible default configuration options that should
work in many cases. However, you can overwrite them via
`keyword arguments <https://docs.python.org/3/glossary.html#term-argument>`_.

Please see :doc:`models` to choose among the supported restoration models.
While training data generation and prediction typically differs among the
models, note that the training process is mostly the same for all models.
For example, a standard (denoising) CARE model can be instantiated via
:class:`csbdeep.models.CARE` and then trained with the
:func:`csbdeep.models.CARE.train` method.
After training, the learned model can be exported via
:func:`csbdeep.models.CARE.export_TF` to be used with our
`Fiji Plugin <https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji>`_.

**Example**

>>> from csbdeep.io import load_training_data
>>> from csbdeep.models import Config, CARE
>>> (X,Y), (X_val,Y_val), axes = load_training_data('my_data.npz', validation_split=0.1)
>>> config = Config(axes)
>>> model = CARE(config, 'my_model')
>>> model.train(X,Y, validation_data=(X_val,Y_val))
>>> model.export_TF()


.. autoclass:: csbdeep.models.Config
    :members:

.. autoclass:: csbdeep.models.CARE
    :members:

Supporting functions:

.. autofunction:: csbdeep.io.load_training_data
.. autofunction:: csbdeep.internals.nets.common_unet
.. autofunction:: csbdeep.internals.train.prepare_model
.. autofunction:: csbdeep.utils.tf.export_SavedModel



.. Advanced topics
.. ---------------

.. .. todo::
..     - ``ReLU`` as last activation → use :func:`csbdeep.train.prepare_model` with (``loss_bg_thresh``, ``loss_bg_thresh``, ``Y``) and :class:`.. csbdeep.train.ParameterDecayCallback` callback...


.. .. note::
..     In principle, we can support other backends than TensorFlow for training, but currently not implemented.
..     Futhermore, we use some TF-specific functions, which are in ``csbdeep.tf``.


Other models
------------

Training other CARE models
(:class:`csbdeep.models.IsotropicCARE`, :class:`csbdeep.models.UpsamplingCARE`)
currently does not differ from that of a
standard model. What changes is the way in which the training data is
generated (see :doc:`datagen`).

.. .. autoclass:: csbdeep.models.IsotropicCARE
..     :members: train
..     :noindex:

.. .. autoclass:: csbdeep.models.UpsamplingCARE
..     :members: train
..     :noindex:
