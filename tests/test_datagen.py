from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import warnings
import numpy as np
import pytest
from tifffile import imread, imsave
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data
from csbdeep.utils import Path, move_image_axes, backend_channels_last



def test_create_patches():
    rng = np.random.RandomState(42)
    def get_data(n_images, axes, shape):
        def _gen():
            for i in range(n_images):
                x = rng.uniform(size=shape)
                y = 5 + 3*x
                yield x, y, axes, None
        return RawData(_gen, n_images, '')

    n_images, n_patches_per_image = 2, 4
    def _create(img_size,img_axes,patch_size,patch_axes):
        X,Y,XYaxes = create_patches (
            raw_data            = get_data(n_images, img_axes, img_size),
            patch_size          = patch_size,
            patch_axes          = patch_axes,
            n_patches_per_image = n_patches_per_image,
        )
        assert len(X) == n_images*n_patches_per_image
        assert np.allclose(X,Y,atol=1e-6)
        if patch_axes is not None:
            assert XYaxes == 'SC'+patch_axes.replace('C','')

    _create((128,128),'YX',(32,32),'YX')
    _create((128,128),'YX',(32,32),None)
    _create((128,128),'YX',(32,32),'XY')
    _create((128,128),'YX',(32,32,1),'XYC')

    _create((32,48,32),'ZYX',(16,32,8),None)
    _create((32,48,32),'ZYX',(16,32,8),'ZYX')
    _create((32,48,32),'ZYX',(16,32,8),'YXZ')
    _create((32,48,32),'ZYX',(16,32,1,8),'YXCZ')



def test_create_save_and_load(tmpdir):
    rng = np.random.RandomState(42)
    tmpdir = Path(str(tmpdir))
    save_file = str(tmpdir / 'data.npz')

    n_images, n_patches_per_image = 2, 4
    def _create(img_size,img_axes,patch_size,patch_axes):
        U,V = (rng.uniform(size=(n_images,)+img_size) for _ in range(2))
        X,Y,XYaxes = create_patches (
            raw_data            = RawData.from_arrays(U,V,img_axes),
            patch_size          = patch_size,
            patch_axes          = patch_axes,
            n_patches_per_image = n_patches_per_image,
            save_file           = save_file
        )
        (_X,_Y), val_data, _XYaxes = load_training_data(save_file,verbose=True)
        assert val_data is None
        assert _XYaxes[-1 if backend_channels_last else 1] == 'C'
        _X,_Y = (move_image_axes(u,fr=_XYaxes,to=XYaxes) for u in (_X,_Y))
        assert np.allclose(X,_X,atol=1e-6)
        assert np.allclose(Y,_Y,atol=1e-6)
        assert set(XYaxes) == set(_XYaxes)
        assert load_training_data(save_file,validation_split=0.5)[2] is not None
        assert all(len(x)==3 for x in load_training_data(save_file,n_images=3)[0])

    _create((  64,64), 'YX',(16,16  ),None)
    _create((  64,64), 'YX',(16,16  ),'YX')
    _create((  64,64), 'YX',(16,16,1),'YXC')
    _create((1,64,64),'CYX',(  16,16),'YX')
    _create((1,64,64),'CYX',(1,16,16),None)
    _create((64,3,64),'YCX',(3,16,16),'CYX')
    _create((64,3,64),'YCX',(16,16,3),'YXC')



def test_rawdata_from_folder(tmpdir):
    rng = np.random.RandomState(42)
    tmpdir = Path(str(tmpdir))

    n_images, img_size, img_axes = 3, (64,64), 'YX'
    data = {'X' : rng.uniform(size=(n_images,)+img_size).astype(np.float32),
            'Y' : rng.uniform(size=(n_images,)+img_size).astype(np.float32)}

    for name,images in data.items():
        (tmpdir/name).mkdir(exist_ok=True)
        for i,img in enumerate(images):
            imsave(str(tmpdir/name/('img_%02d.tif'%i)),img)

    raw_data = RawData.from_folder(str(tmpdir),['X'],'Y',img_axes)
    assert raw_data.size == n_images
    for i,(x,y,axes,mask) in enumerate(raw_data.generator()):
        assert mask is None
        assert axes == img_axes
        assert any(np.allclose(x,u) for u in data['X'])
        assert any(np.allclose(y,u) for u in data['Y'])
