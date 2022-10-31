![example workflow](https://github.com/juglab/n2v/actions/workflows/test_and_deploy.yml/badge.svg)
[![N2V_video](img/n2v_vimeo.png)](https://vimeo.com/305045007)

# Noise2Void - Learning Denoising from Single Noisy Images
Alexander Krull<sup>1,2</sup>, Tim-Oliver Buchholz<sup>2</sup>, Florian Jug</br>
<sup>1</sup><code>a.f.f.krull@bham.ac.uk</code>, <sup>2</sup>Authors contributed equally

The field of image denoising is currently dominated by discriminative deep learning methods that are trained on pairs of noisy input and clean target images. Recently it has been shown that such methods can also be trained without clean targets. Instead, independent pairs of noisy images can be used, in an approach known as NOISE2NOISE (N2N). Here, we introduce NOISE2VOID (N2V), a training scheme that takes this idea one step further. It does not require noisy image pairs, nor clean target images.  Consequently, N2V allows us to train directly on the body of data to be denoised and can therefore be applied when other methods cannot. Especially interesting is the application to biomedical image data, where the acquisition of training targets, clean or noisy, is frequently not possible.  We compare the performance of N2V to approaches that have either clean target images and/or noisy image pairs available. Intuitively, N2V cannot be expected to outperform methods that have more information available during training. Still, we observe that the denoising performance of NOISE2VOID drops in moderation and compares favorably to training-free denoising methods.

Paper: https://arxiv.org/abs/1811.10980

Our implementation is based on [CSBDeep](http://csbdeep.bioimagecomputing.com) ([github](https://github.com/csbdeep/csbdeep)).

# N2V2 - Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a Tweaked Network Architecture
Eva Höck<sup>1,⚹</sup>, Tim-Oliver Buchholz<sup>2,⚹</sup>, Anselm Brachmann<sup>1,⚹</sup>, Florian Jug<sup>3,⁜</sup>, and Alexander Freytag<sup>1,⁜</sup></br>
<sup>1</sup>Carl Zeiss AG, Germany</br>
<sup>2</sup>Facility for Advanced Imaging and Microscopy, Friedrich Miescher Institute for Biomedical Research, Basel, Switzerland</br>
<sup>3</sup>Jug Group, Fondazione Human Technopole, Milano, Italy</br>
<sup>⚹, ⁜</sup>Equal contribution</br>

In recent years, neural network based image denoising approaches have revolutionized the analysis of biomedical microscopy data. Self-supervised methods, such as Noise2Void (N2V), are applicable to virtually all noisy datasets, even without dedicated training data being available. Arguably, this facilitated the fast and widespread adoption of N2V throughout the life sciences. Unfortunately, the blind-spot training underlying N2V can lead to rather visible checkerboard artifacts, thereby reducing the quality of final predictions considerably. In this work, we present two modifications to the vanilla N2V setup that both help to reduce the unwanted artifacts considerably. Firstly, we propose a modified network architecture, i.e. using BlurPool instead of MaxPool layers throughout the used UNet, rolling back the residual-UNet to a non-residual UNet, and eliminating the skip connections at the uppermost UNet level. Additionally, we propose new replacement strategies to determine the pixel intensity values that fill in the elected blind-spot pixels. We validate our modifications on a range of microscopy and natural image data. Based on added synthetic noise from  multiple noise types and at varying amplitudes, we show that both proposed modifications push the current state-of-the-art for fully self-supervised image denoising.

OpenReview: [https://openreview.net/forum?id=IZfQYb4lHVq](https://openreview.net/forum?id=IZfQYb4lHVq)

# Installation
This implementation requires [Tensorflow](https://www.tensorflow.org/install/).
We have tested Noise2Void using Python 3.9 and TensorFlow 2.7 and 2.10.

Note: If you want to use TensorFlow 1.15 you have to install N2V v0.2.1. N2V v0.3.* supports TensorFlow 2 only.

## If you start from scratch...
We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
If you do not yet have a strong opinion, just use it too!

After installing Miniconda, create a conda environment:

```
conda create -n 'n2v' python=3.9
conda activate n2v
```

## Install TensorFlow

The best way to install TensorFLow is to follow the [Tensorflow guidelines](https://www.tensorflow.org/install/pip). 

Note that, after installing TensorFlow, running the following commands in your environment will allow you to use the GPU without having to each 
time run an `export` command (refer to the [Tensorflow step by step](https://www.tensorflow.org/install/pip#linux_1)):
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Option 1: PIP (current stable release)
```
$ pip install n2v
```

## Option 2: Git-Clone and install from sources (current master-branch version)
This option is ideal if you want to edit the code. Clone the repository:

```
$ git clone https://github.com/juglab/n2v.git
```
Change into its directory and install it:

```
$ cd n2v
$ pip install -e .
```
You are now ready to run Noise2Void.

# How to use it?

## Jupyter notebooks
Have a look at our jupyter notebook:
* [2D example BSD68](https://github.com/juglab/n2v/tree/master/examples/2D/denoising2D_BSD68)
* [2D example SEM](https://github.com/juglab/n2v/tree/master/examples/2D/denoising2D_SEM)
* [2D example RGB](https://github.com/juglab/n2v/tree/master/examples/2D/denoising2D_RGB)
* [3D example](https://github.com/juglab/n2v/tree/master/examples/3D)
* [2D StructN2V example synth_mem](https://github.com/juglab/n2v/tree/master/examples/2D/structN2V_2D_synth_mem/)

In order to run the notebooks, install jupyter in your conda environment:
```bash
pip install jupyter
```

Coming soon:
* N2V2 example notebooks.

__Note:__ You can use the N2V2 functionality by providing the following three parameters to the N2V-Config object:
* `blurpool=True`, by default set to `False`
* `skip_skipone=True`, by default set to `False`
* `n2v_manipulator="median"`, by default set to `"uniform_withCP"`
* `unet_residual=False`, by default set to `False`

__Warning:__ Currently, N2V2 does only support 2D data.</br>
__Warning:__ We have not tested N2V2 together with struct-N2V.

## napari

N2V, N2V2 and structN2V are [available in napari](https://www.napari-hub.org/plugins/napari-n2v)!


# How to cite:
```
@inproceedings{krull2019noise2void,
  title={Noise2void-learning denoising from single noisy images},
  author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2129--2137},
  year={2019}
}
N2V2 citation coming soon.
```

see [here](https://github.com/mpicbg-csbd/structured_N2V) for more info on `StructN2V`.

## Note on functional tests

The functional "tests" are meant to be run as regular programs. They are there to make sure that
examples are still running after changes.

## Note on GPU Memory Allocation

In some [cases](https://github.com/juglab/n2v/issues/100) tensorflow is unable to allocate GPU memory and fails. One possible solution could be to set the following environment variable: `export TF_FORCE_GPU_ALLOW_GROWTH=true`
