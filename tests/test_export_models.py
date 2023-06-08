import urllib.request
from pathlib import Path
from zipfile import ZipFile

import pytest

from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    # create temp dir
    tmp = tmp_path_factory.mktemp("data")
    print("Setting up")

    zipPath = Path(tmp, "RGB.zip")
    if not zipPath.exists():
        # download and unzip data
        urllib.request.urlretrieve("https://download.fht.org/jug/n2v/RGB.zip", zipPath)
        with ZipFile(zipPath, "r") as zip_ref:
            zip_ref.extractall(tmp)

    zipPath = Path(tmp, "SEM.zip")
    if not zipPath.exists():
        urllib.request.urlretrieve("https://download.fht.org/jug/n2v/SEM.zip", zipPath)
        with ZipFile(zipPath, "r") as zip_ref:
            zip_ref.extractall(tmp)

    return tmp


@pytest.mark.bioimage_io
def test_model_build_and_export_2D_RGB(temp_dir):
    from bioimageio.core import load_resource_description
    from bioimageio.core.resource_tests import test_model

    str_dir = str(temp_dir)
    str_models = str(Path(temp_dir, "models"))
    model_name = "n2v_2D_RGB"
    bioimage = Path(str_models, model_name, model_name + ".bioimage.io.zip")

    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(
        directory=str_dir, filter="*.png", dims="YXC"
    )
    imgs[0] = imgs[0][..., :3]  # remove alpha
    patches = datagen.generate_patches_from_list(imgs, shape=(64, 64))
    X = patches[:5000]
    X_val = patches[5000:]
    config = N2VConfig(
        X,
        unet_kern_size=3,
        unet_n_first=8,
        unet_n_depth=2,
        train_steps_per_epoch=2,
        train_epochs=2,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=12,
        n2v_perc_pix=5,
        n2v_patch_shape=(64, 64),
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
        train_tensorboard=False,
    )

    # Also set a patch_shape value
    config.patch_shape = config.n2v_patch_shape
    model = N2V(config, model_name, basedir=str_models)
    model.train(X, X_val)
    model.export_TF(
        name="Testing",
        description="PyTest 2D RGB.",
        authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
        test_img=X_val[0],
        axes="YXC",
        patch_shape=(128, 128),
    )

    bioimage_model = load_resource_description(bioimage)
    results = test_model(bioimage_model)

    for entry in results:
        print(entry)
        assert entry["status"] == "passed"


@pytest.mark.bioimage_io
def test_model_build_and_export_2D_SEM(temp_dir):
    from bioimageio.core import load_resource_description
    from bioimageio.core.resource_tests import test_model

    str_dir = str(temp_dir)
    str_models = str(Path(temp_dir, "models"))
    model_name = "n2v_2D_SEM"
    bioimage = Path(str_models, model_name, model_name + ".bioimage.io.zip")

    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory=str_dir, filter="*.tif")
    X = datagen.generate_patches_from_list(imgs[:1], shape=(96, 96))
    X_val = datagen.generate_patches_from_list(imgs[1:], shape=(96, 96))
    config = N2VConfig(
        X,
        unet_kern_size=3,
        train_steps_per_epoch=2,
        train_epochs=2,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=12,
        n2v_perc_pix=1.6,
        n2v_patch_shape=(64, 64),
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
        train_tensorboard=False,
    )

    model = N2V(config, model_name, basedir=str_models)
    model.train(X, X_val)
    model.export_TF(
        name="Testing",
        description="PyTest 2D SEM",
        authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
        test_img=X_val[0, ...],
        axes="YXC",
        patch_shape=(128, 128),
    )
    bioimage_model = load_resource_description(bioimage)
    results = test_model(bioimage_model)

    for entry in results:
        print(entry)
        assert entry["status"] == "passed"
