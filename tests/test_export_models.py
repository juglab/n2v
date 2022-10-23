import pytest
from pathlib import Path

from ruamel.yaml import YAML
import json
import urllib.request
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from zipfile import ZipFile


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    # create temp dir
    tmp = tmp_path_factory.mktemp('data')

    print('Setting up')

    zipPath = Path(tmp, 'RGB.zip')
    if not zipPath.exists():
        # download and unzip data
        urllib.request.urlretrieve('https://download.fht.org/jug/n2v/RGB.zip', zipPath)
        with ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(tmp)

    zipPath = Path(tmp, 'SEM.zip')
    if not zipPath.exists():
        urllib.request.urlretrieve('https://download.fht.org/jug/n2v/SEM.zip', zipPath)
        with ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(tmp)

    return tmp


def test_model_build_and_export_2D_RGB(temp_dir):
    str_dir = str(temp_dir)
    str_models = str(Path(temp_dir, 'models'))
    model_name = 'n2v_2D_RGB'
    model_yaml = 'model.yaml'
    bioimage = Path(str_models, model_name, 'export.bioimage.io.zip')
    extracted = Path(str_models, 'extracted')
    yaml_path = Path(extracted, model_yaml)

    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory=str_dir, filter='*.png', dims='YXC')
    imgs[0] = imgs[0][..., :3]  # remove alpha
    patches = datagen.generate_patches_from_list(imgs, shape=(64, 64))
    X = patches[:5000]
    X_val = patches[5000:]
    config = N2VConfig(X, unet_kern_size=3,
                       unet_n_first=8, unet_n_depth=2, train_steps_per_epoch=2, train_epochs=2, train_loss='mse',
                       batch_norm=True, train_batch_size=12, n2v_perc_pix=5, n2v_patch_shape=(64, 64),
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_tensorboard=False)

    # Also set a patch_shape value
    config.patch_shape = config.n2v_patch_shape
    model = N2V(config, model_name, basedir=str_models)
    model.train(X, X_val)
    model.export_TF(name='Testing',
                    description='PyTest 2D RGB.',
                    authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                    test_img=X_val[0], axes='YXC',
                    patch_shape=(128, 128))

    with ZipFile(bioimage, 'r') as myzip:
        myzip.extract(model_yaml, extracted)
    my_yml = model.get_yml_dict(name='Testing',
                                description='PyTest 2D RGB.',
                                authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                                test_img=X_val[0], axes='YXC',
                                patch_shape=(128, 128))

    yaml = YAML(typ='rt')
    with open(yaml_path, 'r') as infile:
        yml_dict = yaml.load(infile)
    assert my_yml == yml_dict


def test_model_build_and_export_2D_SEM(temp_dir):
    str_dir = str(temp_dir)
    str_models = str(Path(temp_dir, 'models'))
    model_name = 'n2v_2D_SEM'
    model_yaml = 'model.yaml'
    bioimage = Path(str_models, model_name, 'export.bioimage.io.zip')
    extracted = Path(str_models, 'extracted')
    yaml_path = Path(extracted, model_yaml)

    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory=str_dir, filter='*.tif')
    X = datagen.generate_patches_from_list(imgs[:1], shape=(96, 96))
    X_val = datagen.generate_patches_from_list(imgs[1:], shape=(96, 96))
    config = N2VConfig(X, unet_kern_size=3,
                       train_steps_per_epoch=2, train_epochs=2, train_loss='mse', batch_norm=True,
                       train_batch_size=12, n2v_perc_pix=1.6, n2v_patch_shape=(64, 64),
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_tensorboard=False)

    model = N2V(config, model_name, basedir=str_models)
    model.train(X, X_val)
    model.export_TF(name='Testing',
                    description='PyTest 2D SEM',
                    authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                    test_img=X_val[0, ..., 0], axes='YX',
                    patch_shape=(128, 128))
    with ZipFile(bioimage, 'r') as myzip:
        myzip.extract(model_yaml, extracted)
    my_yml = model.get_yml_dict(name='Testing',
                                description='PyTest 2D SEM',
                                authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                                test_img=X_val[0, ..., 0], axes='YX',
                                patch_shape=(128, 128))
    yaml = YAML(typ='rt')
    with open(yaml_path, 'r') as infile:
        yml_dict = yaml.load(infile)
    assert my_yml == yml_dict


def test_export_yaml(tmp_path):
    current = Path(__file__)
    config_path = Path(current.parent, 'test_data/config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    mean_val = []
    mean_val1 = []
    for ele in config['means']:
        mean_val.append(float(ele))
        mean_val1.append(float(ele))
    std_val = []
    std_val1 = []
    for ele in config['stds']:
        std_val.append(float(ele))
        std_val1.append(float(ele))
    axes_val = 'b' + config['axes']
    axes_val = axes_val.lower()
    in_data_range_val = ['-inf', 'inf']
    out_data_range_val = ['-inf', 'inf']
    val = 2 ** config['unet_n_depth']
    min_val = [1, val, val, config['n_channel_in']]
    step_val = [1, val, val, 0]
    halo_val = [0, 22, 22, 0]
    scale_val = [1, 1, 1, 1]
    offset_val = [0, 0, 0, 0]

    yaml = YAML(typ='rt')
    with open(config_path, 'r') as f:
        tr_kwargs_val = yaml.load(f)

    yml_dict = {
        'language': 'python',
        'framework': 'tensorflow',
        'source': 'n2v / denoiseg',
        'inputs': [{
            'name': 'inputs',
            'axes': 'axes_val',
            'data_type': 'float32',
            'data_range': in_data_range_val,
            'shape': {
                'min': 'min_val',
                'step': 'step_val'
            }
        }],
        'outputs': [{
            'name': 'placeholder',
            'axes': 'axes_val',
            'data_type': 'float32',
            'data_range': out_data_range_val,
            'halo': halo_val,
            'shape': {
                'scale': scale_val,
                'offset': offset_val
            }
        }],
        'training': {
            'source': 'n2v.train()',
            'kwargs': tr_kwargs_val
        },
        'prediction': {
            'preprocess': {
                'kwargs': {
                    'mean': mean_val,
                    'stdDev': std_val
                }
            },
            'postprocess': {
                'kwargs': {
                    'mean': mean_val1,
                    'stdDev': std_val1
                }
            }
        }
    }

    ymal_path_0 = Path(tmp_path, 'test.yml')
    ymal_path_1 = Path(tmp_path, 'test1.yml')

    yaml.default_flow_style = False
    with open(ymal_path_0, 'w+') as outfile:
        yaml.dump(yml_dict, outfile)

    yaml1 = YAML(typ='rt')
    yaml1.default_flow_style = False
    with open(ymal_path_1, 'w+') as fout:
        yaml1.dump(yml_dict, fout)

    with open(ymal_path_1, 'r') as infile:
        read_yml = yaml.load(infile)
    assert read_yml == yml_dict
