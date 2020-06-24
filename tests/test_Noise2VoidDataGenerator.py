from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import urllib.request
import os
import zipfile


def test_generate_patches_2D():

    if not os.path.isdir('data'):
        os.mkdir('data')
    zip_path = "data/RGB.zip"
    if not os.path.isfile(zip_path):
        data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/Frru2hsjjAljpfW/download', zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')

    datagen = N2V_DataGenerator()

    imgs = datagen.load_imgs_from_directory(directory="data", filter='*.png', dims='YXC')
    imgs[0] = imgs[0][..., :3]
    patches = datagen.generate_patches_from_list(imgs, shape=(1100, 2800))
    assert len(patches) == 1
    patches = datagen.generate_patches_from_list(imgs, shape=(550, 1400))
    assert len(patches) == 4
    patches = datagen.generate_patches_from_list(imgs, shape=(110, 280))
    assert len(patches) == 100

def test_generate_patches_3D():

    if not os.path.isdir('data'):
        os.mkdir('data')
    zip_path = 'data/flywing-data.zip'
    if not os.path.isfile(zip_path):
        # download and unzip data
        data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/RKStdwKo4FlFrxE/download', zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')

    datagen = N2V_DataGenerator()

    imgs = datagen.load_imgs_from_directory(directory="data", filter='flywing.tif', dims='ZYX')
    print(imgs[0].shape)
    patches = datagen.generate_patches_from_list(imgs[:1], shape=(35, 520, 692))
    assert len(patches) == 1
    patches = datagen.generate_patches_from_list(imgs[:1], shape=(5, 52, 174))
    assert len(patches) == 210

