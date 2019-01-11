CSBDeep – a toolbox for CARE
============================

.. image:: https://badge.fury.io/py/csbdeep.svg
   :target: https://pypi.org/project/csbdeep
   :alt: PyPI version

.. image:: https://travis-ci.com/CSBDeep/CSBDeep.svg?branch=master
   :target: https://travis-ci.com/CSBDeep/CSBDeep
   :alt: Linux build status

.. image:: https://ci.appveyor.com/api/projects/status/xbl32vudixshj990/branch/master?svg=true
   :target: https://ci.appveyor.com/project/UweSchmidt/csbdeep-c2jtk
   :alt: Windows build status

This is the documentation for the
`CSBDeep Python package <https://github.com/csbdeep/csbdeep>`_,
which provides a toolbox for content-aware restoration (CARE)
of (fluorescence) microscopy images, based on deep learning via
`Keras <https://keras.io/>`_ and `TensorFlow <https://www.tensorflow.org/>`_.
Please see the
`CSBDeep website <http://csbdeep.bioimagecomputing.com/>`_
for more information with links to our manuscript and supplementary material.

After :doc:`installation </install>` of the Python package,
we recommend to follow the provided `examples`_
that provide step-by-step instructions via `Jupyter <http://jupyter.org/>`_ notebooks
on how to use this package.

.. _`examples`: http://csbdeep.bioimagecomputing.com/examples

.. note::
    This is an early version of the software.
    Several features are still missing, including the use of network
    ensembles and the application for surface projection.
    Furthermore, creating training data via simulation is currently
    not implemented.


Table of contents
-----------------

.. toctree::
   install
   models
   datagen
   training
   prediction


.. **Last updated:** |today|
