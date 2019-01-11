Training data generation
========================

The details of training data generation vary depending on the
intended image restoration task (see :doc:`models`).
We recommend to start with one of our `examples`_.

.. _`examples`: http://csbdeep.bioimagecomputing.com/examples/

We first explain the process for a standard CARE model.
To that end, we need to specify matching pairs of raw *source* and *target* images,
which is done with a :class:`csbdeep.data.RawData` object.
It is important that you correctly set the *axes* of the raw images, e.g. to
``CYX`` for 2D images with a channel dimension before the two lateral dimensions.

Image Axes
    ``X``: columns,
    ``Y``: rows,
    ``Z``: planes,
    ``C``: channels,
    ``T``: frames/time,
    (``S``: samples/images)

.. note::
    - The raw data should be representative of all images that the CARE network
      will potentially be applied to after training.
    - Usually, it is best to not process the raw images in any other way (e.g. deconvolution).
    - Source and target images must be well-aligned to obtain effective CARE networks.

.. autoclass:: csbdeep.data.RawData
    :members:


------


With the raw data specified as above, the function :func:`csbdeep.data.create_patches`
can be used to randomly extract patches of a given size that are suitable for training.
By default, patches are normalized based on a range of percentiles
computed on the raw images, which tends to lead to more robust CARE networks in our experience.
If not specified otherwise, patches which are purely background are also excluded
from being extracted, since they do not contain interesting structures.

.. autofunction:: csbdeep.data.create_patches

Supporting functions:

.. autofunction:: csbdeep.data.no_background_patches
.. autofunction:: csbdeep.data.norm_percentiles
.. autofunction:: csbdeep.data.sample_percentiles
.. autofunction:: csbdeep.io.save_training_data




Anisotropic distortions
-----------------------

We provide the function :func:`csbdeep.data.anisotropic_distortions`
that returns a :class:`csbdeep.data.Transform` object (see `Transforms`_)
to be used for creating training data for
:class:`csbdeep.models.UpsamplingCARE` and
:class:`csbdeep.models.IsotropicCARE`.

.. autofunction:: csbdeep.data.anisotropic_distortions




Transforms
----------

A :class:`csbdeep.data.Transform` can be used to modify and augment the set of raw images
before they are being use in :func:`csbdeep.data.create_patches` to generate training data.

.. autoclass:: csbdeep.data.Transform
    :members:


Data augmention
^^^^^^^^^^^^^^^

Instead of recording raw images where structures of interest appear in all
possible appearance variations, it can be easier to augment the raw dataset by
including some of those variations that can be easily synthesized. Typical
examples are axis-aligned rotations if structures of interest can appear at
arbitrary rotations. We currently haven't implemented any such
transformations, but plan to at least add axis-aligned rotations and flips
later.


Other Transforms
^^^^^^^^^^^^^^^^

.. automodule:: csbdeep.data
   :members: permute_axes, crop_images
