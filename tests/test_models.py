from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from itertools import product

# import warnings
import numpy as np
import pytest
from csbdeep.data import NoNormalizer, NoResizer
from csbdeep.internals.predict import tile_overlap
from keras import backend as K

from csbdeep.internals.nets import receptive_field_unet
from csbdeep.models import Config, CARE
from csbdeep.utils import axes_dict
from csbdeep.utils.six import FileNotFoundError



def config_generator(**kwargs):
    assert 'axes' in kwargs
    keys, values = kwargs.keys(), kwargs.values()
    values = [v if isinstance(v,(list,tuple)) else [v] for v in values]
    for p in product(*values):
        yield Config(**dict(zip(keys,p)))



def test_config():
    assert K.image_data_format() in ('channels_first','channels_last')
    def _with_channel(axes):
        axes = axes.upper()
        if 'C' in axes:
            return axes
        return (axes+'C') if K.image_data_format() == 'channels_last' else ('C'+axes)

    axes_list = [
        ('yx',_with_channel('YX')),
        ('ytx',_with_channel('YTX')),
        ('zyx',_with_channel('ZYX')),
        ('YX',_with_channel('YX')),
        ('XYZ',_with_channel('XYZ')),
        ('XYT',_with_channel('XYT')),
        ('SYX',_with_channel('YX')),
        ('SXYZ',_with_channel('XYZ')),
        ('SXTY',_with_channel('XTY')),
        (_with_channel('YX'),_with_channel('YX')),
        (_with_channel('XYZ'),_with_channel('XYZ')),
        (_with_channel('XTY'),_with_channel('XTY')),
        (_with_channel('SYX'),_with_channel('YX')),
        (_with_channel('STYX'),_with_channel('TYX')),
        (_with_channel('SXYZ'),_with_channel('XYZ')),
    ]

    for (axes,axes_ref) in axes_list:
        assert Config(axes).axes == axes_ref

    with pytest.raises(ValueError):
        Config('XYC')
        Config('CXY')
    with pytest.raises(ValueError):
        Config('XYZC')
        Config('CXYZ')
    with pytest.raises(ValueError):
        Config('XTYC')
        Config('CXTY')
    with pytest.raises(ValueError): Config('XYZT')
    with pytest.raises(ValueError): Config('tXYZ')
    with pytest.raises(ValueError): Config('XYS')
    with pytest.raises(ValueError): Config('XSYZ')



@pytest.mark.parametrize('config', config_generator(
    axes                  = ['YX','ZYX'],
    n_channel_in          = [1,2],
    n_channel_out         = [1,2],
    probabilistic         = [False,True],
    unet_residual         = [False,True],
    unet_n_depth          = [1,2],
    # unet_kern_size        = [3],
    # unet_n_first          = [32],
    # unet_last_activation  = ['linear'],
    # unet_input_shape      = [(None, None, 1)],
    #
    # train_batch_size      = [16],
    # train_checkpoint      = ['weights_best.h5'],
    # train_epochs          = [100],
    # train_learning_rate   = [0.0004],
    # train_loss            = ['mae'],
    # train_reduce_lr       = [{'factor': 0.5, 'patience': 10}],
    # train_steps_per_epoch = [400],
    # train_tensorboard     = [True],
))
def test_model_build_and_export(tmpdir,config):
    K.clear_session()
    def _build():
        with pytest.raises(FileNotFoundError):
            CARE(None,basedir=str(tmpdir))

        CARE(config,name='model',basedir=None)
        with pytest.raises(ValueError):
            CARE(None,basedir=None)

        CARE(config,basedir=str(tmpdir)).export_TF()

        with pytest.warns(UserWarning):
            CARE(config,name='model',basedir=str(tmpdir))
            CARE(config,name='model',basedir=str(tmpdir))
            CARE(None,name='model',basedir=str(tmpdir))
    if config.is_valid():
        _build()
    else:
        with pytest.raises(ValueError):
            _build()



@pytest.mark.parametrize('config', filter(lambda c: c.is_valid(), config_generator(
    axes                  = ['YX','ZYX'],
    n_channel_in          = [1,2],
    n_channel_out         = [1,2],
    probabilistic         = [False,True],
    # unet_residual         = [False,True],
    unet_n_depth          = [1],

    unet_kern_size        = [3],
    unet_n_first          = [4],
    unet_last_activation  = ['linear'],
    # unet_input_shape      = [(None, None, 1)],

    train_loss            = ['mae','laplace'],
    train_epochs          = [2],
    train_steps_per_epoch = [2],
    # train_learning_rate   = [0.0004],
    train_batch_size      = [2],
    # train_tensorboard     = [True],
    # train_checkpoint      = ['weights_best.h5'],
    # train_reduce_lr       = [{'factor': 0.5, 'patience': 10}],
)))
def test_model_train(tmpdir,config):
    rng = np.random.RandomState(42)
    K.clear_session()
    X = rng.uniform(size=(4,)+(32,)*config.n_dim+(config.n_channel_in,))
    Y = rng.uniform(size=(4,)+(32,)*config.n_dim+(config.n_channel_out,))
    model = CARE(config,basedir=str(tmpdir))
    model.train(X,Y,(X,Y))



@pytest.mark.parametrize('config', filter(lambda c: c.is_valid(), config_generator(
    axes                  = ['YX','ZYX'],
    n_channel_in          = [1,2],
    n_channel_out         = [1,2],
    probabilistic         = [False,True],
    # unet_residual         = [False,True],
    unet_n_depth          = [2],

    unet_kern_size        = [3],
    unet_n_first          = [4],
    unet_last_activation  = ['linear'],
    # unet_input_shape      = [(None, None, 1)],
)))
def test_model_predict(tmpdir,config):
    rng = np.random.RandomState(42)
    normalizer, resizer = NoNormalizer(), NoResizer()

    K.clear_session()
    model = CARE(config,basedir=str(tmpdir))
    axes = config.axes

    def _predict(imdims,axes):
        img = rng.uniform(size=imdims)
        # print(img.shape, axes, config.n_channel_out)
        mean, scale = model._predict_mean_and_scale(img, axes, normalizer, resizer)
        if config.probabilistic:
            assert mean.shape == scale.shape
        else:
            assert scale is None

        if 'C' not in axes:
            if config.n_channel_out == 1:
                assert mean.shape == img.shape
            else:
                assert mean.shape == img.shape + (config.n_channel_out,)
        else:
            channel = axes_dict(axes)['C']
            imdims[channel] = config.n_channel_out
            assert mean.shape == tuple(imdims)


    imdims = list(rng.randint(20,40,size=config.n_dim))
    div_n = 2**config.unet_n_depth
    imdims = [(d//div_n)*div_n for d in imdims]

    if config.n_channel_in == 1:
        _predict(imdims,axes=axes.replace('C',''))

    channel = rng.randint(0,config.n_dim)
    imdims.insert(channel,config.n_channel_in)
    _axes = axes.replace('C','')
    _axes = _axes[:channel]+'C'+_axes[channel:]
    _predict(imdims,axes=_axes)



@pytest.mark.parametrize('config', filter(lambda c: c.is_valid(), config_generator(
    axes                  = ['YX','ZYX'],
    n_channel_in          = [1],
    n_channel_out         = [1],
    probabilistic         = [False],
    # unet_residual         = [False,True],
    unet_n_depth          = [1,2,3],
    unet_kern_size        = [3,5],

    unet_n_first          = [4],
    unet_last_activation  = ['linear'],
    # unet_input_shape      = [(None, None, 1)],
)))
def test_model_predict_tiled(tmpdir,config):
    """
    Test that tiled prediction yields the same
    or similar result as compared to predicting
    the whole image at once.
    """
    rng = np.random.RandomState(42)
    normalizer, resizer = NoNormalizer(), NoResizer()

    K.clear_session()
    model = CARE(config,basedir=str(tmpdir))

    def _predict(imdims,axes,n_tiles):
        img = rng.uniform(size=imdims)
        # print(img.shape, axes)
        mean,       scale       = model._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles=None)
        mean_tiled, scale_tiled = model._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles=n_tiles)
        assert mean.shape == mean_tiled.shape
        if config.probabilistic:
            assert scale.shape == scale_tiled.shape
        error_max = np.max(np.abs(mean-mean_tiled))
        # print('n, k, err = {0}, {1}x{1}, {2}'.format(model.config.unet_n_depth, model.config.unet_kern_size, error_max))
        assert error_max < 1e-3
        return mean, mean_tiled

    imdims = list(rng.randint(50,70,size=config.n_dim))
    if config.n_dim == 3:
        imdims[0] = 16 # make one dim small, otherwise test takes too long
    div_n = 2**config.unet_n_depth
    imdims = [(d//div_n)*div_n for d in imdims]

    imdims.insert(0,config.n_channel_in)
    axes = 'C'+config.axes.replace('C','')

    for n_tiles in (
        -1, 1.2,
        [1]+[1.2]*config.n_dim,
        [1]*config.n_dim, # missing value for channel axis
        [2]+[1]*config.n_dim, # >1 tiles for channel axis
    ):
        with pytest.raises(ValueError):
            _predict(imdims,axes,n_tiles)

    for n_tiles in [list(rng.randint(1,5,size=config.n_dim)) for _ in range(3)]:
        # print(imdims,axes,[1]+n_tiles)
        if config.n_channel_in == 1:
            _predict(imdims[1:],axes[1:],n_tiles)
        _predict(imdims,axes,[1]+n_tiles)

    # legacy api: tile only largest dimension
    n_blocks = np.max(imdims) // div_n
    for n_tiles in (2,5,n_blocks+1):
        with pytest.warns(UserWarning):
            if config.n_channel_in == 1:
                _predict(imdims[1:],axes[1:],n_tiles)
            _predict(imdims,axes,n_tiles)



@pytest.mark.parametrize('n_depth', (1,2,3,4,5))
@pytest.mark.parametrize('kern_size', (3,5,7))
def test_tile_overlap(n_depth, kern_size):
    K.clear_session()
    img_size = 1280
    rf_x, rf_y = receptive_field_unet(n_depth,kern_size,2,img_size)
    assert rf_x == rf_y
    rf = rf_x
    assert np.abs(rf[0]-rf[1]) < 10
    assert sum(rf)+1 < img_size
    assert max(rf) == tile_overlap(n_depth,kern_size)
    # print("receptive field of n_depth %d and kernel size %d: %s"%(n_depth,kern_size,rf));
