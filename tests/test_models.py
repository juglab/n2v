import pytest
import yaml
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import urllib.request
import os
from zipfile import ZipFile
import shutil

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
            unet_n_first=64, unet_n_depth=3, train_steps_per_epoch=2, train_epochs=2, train_loss='mse',
            batch_norm=True, train_batch_size=128, n2v_perc_pix=5, n2v_patch_shape=(64, 64),
            n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)        
        # Also set a patch_shape value
        config.patch_shape=config.n2v_patch_shape
        model_name = 'n2v_2D_RGB'
        basedir = 'models'
        model = N2V(config, model_name, basedir=basedir)
        model.train(X, X_val)
        model.export_TF()
        with ZipFile('models/n2v_2D_RGB/export.modelzoo.zip', 'r') as myzip:
            myzip.extract('config.yml', 'models/n2v_2D_RGB/extracted')
        my_yml = model.get_yml_dict(patch_shape=config.n2v_patch_shape)
        with open('models/n2v_2D_RGB/extracted/config.yml', 'r') as infile:
            yml_dict = yaml.load(infile)
        assert my_yml == yml_dict
        
    def test_model_build_and_export_2D_SEM(self):
        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory = "data", filter='*.tif')
        X = datagen.generate_patches_from_list(imgs[:1], shape=(96,96))
        X_val = datagen.generate_patches_from_list(imgs[1:], shape=(96,96))
        config = N2VConfig(X, unet_kern_size=3,
                   train_steps_per_epoch=2,train_epochs=2, train_loss='mse', batch_norm=True,
                   train_batch_size=128, n2v_perc_pix=1.6, n2v_patch_shape=(64, 64),
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)

        model_name = 'n2v_2D_SEM'
        basedir = 'models'
        model = N2V(config, model_name, basedir=basedir)
        model.train(X, X_val)
        model.export_TF()
        with ZipFile('models/n2v_2D_SEM/export.modelzoo.zip', 'r') as myzip:
            myzip.extract('config.yml', 'models/n2v_2D_SEM/extracted')
        my_yml = model.get_yml_dict()
        with open('models/n2v_2D_SEM/extracted/config.yml', 'r') as infile:
            yml_dict = yaml.load(infile)
        assert my_yml == yml_dict