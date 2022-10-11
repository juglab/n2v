[![](https://travis-ci.com/juglab/n2v.svg?branch=master)](https://travis-ci.com/juglab/n2v)
[![N2V_video](img/n2v_vimeo.png)](https://vimeo.com/305045007)

# Noise2Void - Learning Denoising from Single Noisy Images
Alexander Krull<sup>1,2</sup>, Tim-Oliver Buchholz<sup>2</sup>, Florian Jug</br>
<sup>1</sup><code>krull@mpi-cbg.de</code>, <sup>2</sup>Authors contributed equally

The field of image denoising is currently dominated by discriminative deep learning methods that are trained on pairs of noisy input and clean target images. Recently it has been shown that such methods can also be trained without clean targets. Instead, independent pairs of noisy images can be used, in an approach known as NOISE2NOISE (N2N). Here, we introduce NOISE2VOID (N2V), a training scheme that takes this idea one step further. It does not require noisy image pairs, nor clean target images.  Consequently, N2V allows us to train directly on the body of data to be denoised and can therefore be applied when other methods cannot. Especially interesting is the application to biomedical image data, where the acquisition of training targets, clean or noisy, is frequently not possible.  We compare the performance of N2V to approaches that have either clean target images and/or noisy image pairs available. Intuitively, N2V cannot be expected to outperform methods that have more information available during training. Still, we observe that the denoising performance of NOISE2VOID drops in moderation and compares favorably to training-free denoising methods.

Paper: https://arxiv.org/abs/1811.10980

Our implementation is based on [CSBDEEP](http://csbdeep.bioimagecomputing.com) ([github](https://github.com/csbdeep/csbdeep)).

## Installation
This implementation requires [Tensorflow](https://www.tensorflow.org/install/).
We have tested Noise2Void using Python 3.9 and TensorFlow 2.7 and 2.10.

Note: If you want to use TensorFlow 1.15 you have to install N2V v0.2.1. N2V v0.3.* supports TensorFlow 2 only.

#### If you start from scratch...
We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
If you do not yet have a strong opinion, just use it too!

After installing Miniconda, create a conda environment:

```
conda create -n 'n2v' python=3.9
conda activate n2v
```

#### Install TensorFlow

The best way to install is to follow the [Tensorflow guidelines](https://www.tensorflow.org/install/pip). 

Note that running the following commands in your environment will allow you to use the GPU without having to each 
time run an `export` command (refer to the [Tensorflow step by step](https://www.tensorflow.org/install/pip#linux_1)):
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

#### Option 1: PIP (current stable release)
```
$ pip install n2v
```

#### Option 2: Git-Clone and install from sources (current master-branch version)
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

## How to use it?
Have a look at our jupyter notebook:
* [2D example BSD68](https://github.com/juglab/n2v/tree/master/examples/2D/denoising2D_BSD68)
* [2D example SEM](https://github.com/juglab/n2v/tree/master/examples/2D/denoising2D_SEM)
* [2D example RGB](https://github.com/juglab/n2v/tree/master/examples/2D/denoising2D_RGB)
* [3D example](https://github.com/juglab/n2v/tree/master/examples/3D)
* [2D StructN2V example synth_mem](https://github.com/juglab/n2v/tree/master/examples/2D/structN2V_2D_synth_mem/)

## How to cite:
```
@inproceedings{krull2019noise2void,
  title={Noise2void-learning denoising from single noisy images},
  author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2129--2137},
  year={2019}
}
```

see [here](https://github.com/mpicbg-csbd/structured_N2V) for more info on `StructN2V`.

## Note on functional tests

The functional "tests" are meant to be run as regular programs. They are there to make sure that
examples are still running after changes.

## Note on GPU Memory Allocation

In some [cases](https://github.com/juglab/n2v/issues/100) tensorflow is unable to allocate GPU memory and fails. One possible solution could be to set the following environment variable: `export TF_FORCE_GPU_ALLOW_GROWTH=true`
