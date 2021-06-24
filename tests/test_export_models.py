import pytest
from ruamel.yaml import YAML
import itertools
import json
import numpy as np
import shutil
import os
import sys
import urllib.request
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from zipfile import ZipFile

class TestExportModel():
    

    
    def setup_class(self):
        print('Setting up')
        # create a folder for our data
        if not os.path.isdir('data'):
            os.mkdir('data')
        zipPath = "data/RGB.zip"
        if not os.path.isfile(zipPath):
            # download and unzip data
            urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/Frru2hsjjAljpfW/download', zipPath)
            with ZipFile(zipPath, 'r') as zip_ref:
                zip_ref.extractall('data')
                
        zipPath = "data/SEM.zip"
        if not os.path.isfile(zipPath):
            urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/pXgfbobntrw06lC/download', zipPath)
            with ZipFile(zipPath, 'r') as zip_ref:
                zip_ref.extractall('data')
                
        # Clean results directory        
        if os.path.exists('models'): 
            shutil.rmtree('models')
                
    def test_model_build_and_export_2D_RGB(self):
        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory="data", filter='*.png', dims='YXC')
        imgs[0] = imgs[0][...,:3]
        patches = datagen.generate_patches_from_list(imgs, shape=(64,64))
        X = patches[:5000]
        X_val = patches[5000:]
        config = N2VConfig(X, unet_kern_size=3,
            unet_n_first=8, unet_n_depth=2, train_steps_per_epoch=2, train_epochs=2, train_loss='mse',
            batch_norm=True, train_batch_size=12, n2v_perc_pix=5, n2v_patch_shape=(64, 64),
            n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_tensorboard=False)
        # Also set a patch_shape value
        config.patch_shape=config.n2v_patch_shape
        model_name = 'n2v_2D_RGB'
        basedir = 'models'
        model = N2V(config, model_name, basedir=basedir)
        model.train(X, X_val)
        model.export_TF(name='Testing', 
                description='PyTest 2D RGB.', 
                authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                test_img=X_val[0], axes='YXC',
                patch_shape=(128,128))
        with ZipFile('models/n2v_2D_RGB/export.bioimage.io.zip', 'r') as myzip:
            myzip.extract('model.yaml', 'models/n2v_2D_RGB/extracted')
        my_yml = model.get_yml_dict(name='Testing', 
                description='PyTest 2D RGB.', 
                authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                test_img=X_val[0], axes='YXC',
                patch_shape=(128,128))
        
        yaml=YAML(typ='rt') 
        with open('models/n2v_2D_RGB/extracted/model.yaml', 'r') as infile:
            yml_dict = yaml.load(infile)
        assert my_yml == yml_dict
        
    def test_model_build_and_export_2D_SEM(self):
        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory = "data", filter='*.tif')
        X = datagen.generate_patches_from_list(imgs[:1], shape=(96,96))
        X_val = datagen.generate_patches_from_list(imgs[1:], shape=(96,96))
        config = N2VConfig(X, unet_kern_size=3,
                   train_steps_per_epoch=2,train_epochs=2, train_loss='mse', batch_norm=True,
                   train_batch_size=12, n2v_perc_pix=1.6, n2v_patch_shape=(64, 64),
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_tensorboard=False)

        model_name = 'n2v_2D_SEM'
        basedir = 'models'
        model = N2V(config, model_name, basedir=basedir)
        model.train(X, X_val)
        model.export_TF(name='Testing', 
                description='PyTest 2D SEM', 
                authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                test_img=X_val[0,...,0], axes='YX',
                patch_shape=(128,128))
        with ZipFile('models/n2v_2D_SEM/export.bioimage.io.zip', 'r') as myzip:
            myzip.extract('model.yaml', 'models/n2v_2D_SEM/extracted')
        my_yml = model.get_yml_dict(name='Testing', 
                description='PyTest 2D SEM', 
                authors=["Gabriella Turek", "Tim-Oliver Buchholz"],
                test_img=X_val[0,...,0], axes='YX',
                patch_shape=(128,128))
        yaml=YAML(typ='rt')
        with open('models/n2v_2D_SEM/extracted/model.yaml', 'r') as infile:
            yml_dict = yaml.load(infile)
        assert my_yml == yml_dict
        
    def test_export_yaml(self):
        with open('test_data/config.json','r') as f:
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
        val = 2**config['unet_n_depth']
        min_val = [1, val, val, config['n_channel_in']]
        step_val = [1, val, val, 0]
        halo_val = [0, 22,22, 0]
        scale_val = [1, 1, 1, 1]
        offset_val = [0, 0, 0, 0]
        
        yaml=YAML(typ='rt')
        with open('test_data/config.json','r') as f:
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
        
        yaml.default_flow_style=False
        with open('test.yml', 'w+') as outfile:
            yaml.dump(yml_dict, outfile)
            
        yaml1=YAML(typ='rt')
        yaml1.default_flow_style=False
        with open('test1.yml', 'w+') as fout:
            yaml1.dump(yml_dict, fout)

        with open('test.yml', 'r') as infile:
            read_yml = yaml.load(infile)
        assert read_yml == yml_dict