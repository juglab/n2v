Model application
=================

After :doc:`training a CARE model <training>`,
we can apply it to raw images that we want to restore.

.. note::
    Alternatively, you can call :func:`csbdeep.models.CARE.export_TF`
    to export the model and use it with our
    `Fiji Plugin <https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji>`_.

We first create a CARE model with the same name that we have previously
used for training it. By not providing a configuration (``config = None``), it will
automatically be loaded from the model's folder. Furthermore, the model's weights (parameters)
are automatically loaded. However, you can also manually load
specific weights by invoking :func:`csbdeep.models.CARE.load_weights`.
Finally, the model can be applied to a raw image with :func:`predict`,
using sensible raw image preparation by default (see `Image preparation`_
for details and options).


**Examples**

>>> from tifffile import imread
>>> from csbdeep.models import CARE
>>> model = CARE(config=None, name='my_model')
>>> x = imread('my_image.tif')
>>> restored = model.predict(x, axes='YX')

>>> from tifffile import imread
>>> from csbdeep.models import IsotropicCARE
>>> model = CARE(config=None, name='my_model')
>>> x = imread('my_image.tif')
>>> restored = model.predict(x, axes='ZYX', factor=4)



.. autoclass:: csbdeep.models.CARE
    :noindex:
    :members: predict, predict_probabilistic, load_weights

.. autoclass:: csbdeep.models.IsotropicCARE
    :members:

.. autoclass:: csbdeep.models.UpsamplingCARE
    :members:

Return type for probabilistic prediction:

.. autoclass:: csbdeep.internals.probability.ProbabilisticPrediction
    :members:


Image preparation
---------------------

Before a CARE model can be applied to a raw input image, we first need to specify
image normalization_ and resizing_ methods. By default, we use
:class:`csbdeep.predict.PercentileNormalizer` with sensible low and high
percentile values (compatible to the defaults used for :doc:`training
data generation <datagen>`).
Furthermore, although a CARE model can be applied to various image
sizes after training, some image dimensions must be divisible by powers of
two, depending on the *depth* of the neural network. To that end, we recommend
to use :class:`csbdeep.predict.PadAndCropResizer`, which, if necessary, will
enlarge the image by a few pixels before prediction and remove those
additional pixels afterwards, such that the size of the raw input image is
retained.


Normalization
^^^^^^^^^^^^^

All normalization methods must subclass :class:`csbdeep.data.Normalizer`.

.. autoclass:: csbdeep.data.Normalizer
    :members:

.. autoclass:: csbdeep.data.NoNormalizer

.. autoclass:: csbdeep.data.PercentileNormalizer
    :members:


Resizing
^^^^^^^^

All resizing methods must subclass :class:`csbdeep.data.Resizer`.

.. autoclass:: csbdeep.data.Resizer
    :members:

.. autoclass:: csbdeep.data.NoResizer

.. autoclass:: csbdeep.data.PadAndCropResizer
    :members:
