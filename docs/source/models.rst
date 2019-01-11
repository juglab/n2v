Model overview
==============

This is an overview of the currently supported restoration
models that are tailored to commonly-used imaging scenarios:


:class:`csbdeep.models.CARE`
----------------------------

  Description:
    - Standard model that learns a mapping from input (degraded) to output (restored) images.
    - Input/output can be (multi-channel) 2D or 3D stacks.
    - Expects spatially registered input/output pairs.

  Typical use-case:
    - Denoising of live-cell images (e.g. acquired with reduced laser power/exposure).
    - Improving SNR of fast time-lapses (e.g. of vesicle trafficking).

  Examples:
    - `3D denoising <http://csbdeep.bioimagecomputing.com/examples/denoising3D>`_
    - `2D denoising (probabilistic) <http://csbdeep.bioimagecomputing.com/examples/denoising2D_probabilistic>`_


:class:`csbdeep.models.UpsamplingCARE`
--------------------------------------

  Description:
    - Extension of the standard model that will additionally increase sampling along a given (e.g. axial) dimension by a given factor ``s``.
    - Input/output pairs should be registered 3D stacks with the desired pixel size.
    - After training, the model is applied to lower-resolution data producing output stacks with an ``s``-fold increased number of sample planes.

  Typical use-case:
    - Improving the axial resolution of volumetric time-lapses when only a limited number of focal planes can be acquired.

  Examples:
    - `3D upsampling <http://csbdeep.bioimagecomputing.com/examples/upsampling3D>`_


:class:`csbdeep.models.IsotropicCARE`
-------------------------------------

  Description:
    - Model that improves axial resolution of (axially) anisotropic stacks.
    - Takes anisotropic 3D stacks as input (important: doesn't need corresponding output stacks).
    - The PSF of the microscope has to be (approximately) known.
    - Assumes isotropic distribution of biological structures (i.e. don't use it with highly anisotropic tissue like cortical tissue).

  Typical use-case:
    - Enhancing axial resolution of (already acquired) light-sheet microscopy time-lapses of developing embryos.

  Examples:
    - `Isotropic reconstruction <http://csbdeep.bioimagecomputing.com/examples/isotropic_reconstruction>`_
