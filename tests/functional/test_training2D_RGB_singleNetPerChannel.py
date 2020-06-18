#!/usr/bin/env python
# Noise2Void - 2D Example for RGB Data"
# Using TensorFlow backend
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib.request
import os
import zipfile

# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')
# check if data has been downloaded already
zipPath = "data/RGB.zip"
if not os.path.exists(zipPath):
    # download and unzip data
    data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/Frru2hsjjAljpfW/download', zipPath)
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall("data")

# For training, we will load __one__ low-SNR RGB image and use the <code>N2V_DataGenerator</code> to extract non-overlapping patches
datagen = N2V_DataGenerator()

# We will load all the '.png' files from the 'data' directory. In our case it is only one.
# The function will return a list of images (numpy arrays).
# In the 'dims' parameter we specify the order of dimensions in the image files we are reading:
# 'C' stands for channels (color)
imgs = datagen.load_imgs_from_directory(directory="./data", filter='*.png', dims='YXC')

print('shape of loaded images: ',imgs[0].shape)
# Remove alpha channel
imgs[0] = imgs[0][...,:3]
print('shape without alpha:    ',imgs[0].shape)

# We have to remove the added extra dimension to display it as 2D image
#plt.figure(figsize=(32,16))
#plt.imshow(imgs[0][0,:,1000:3000,...])
#plt.show()

# Extract patches for training and validation
# The parameter 'shape' defines the size of these patches
patches = datagen.generate_patches_from_list(imgs, shape=(64,64))

# Patches are created so they do not overlap.
# (Note: this is not the case if you specify a number of patches. See the docstring for details!)
# Non-overlapping patches enable us to split them into a training and validation set.
X = patches[:5000]
X_val = patches[5000:]

#plt.figure(figsize=(14,7))
#plt.subplot(1,2,1)
#plt.imshow(X[0,...])
#plt.title('Training Patch')
#plt.subplot(1,2,2)
#plt.imshow(X_val[0,...])
#plt.title('Validation Patch')
#plt.show()

# You can increase "train_steps_per_epoch" to get even better results at the price of longer computation
config = N2VConfig(X, unet_kern_size=3,
            unet_n_first=64, unet_n_depth=3, train_steps_per_epoch=5, train_epochs=25, train_loss='mse',
            batch_norm=True, train_batch_size=128, n2v_perc_pix=5, n2v_patch_shape=(64, 64),
            n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=True)
vars(config)

# name used to identify the model
model_name = 'n2v_2D_RGB'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model
model = N2V(config, model_name, basedir=basedir)
history = model.train(X, X_val)
print(sorted(list(history.history.keys())))
#plt.figure(figsize=(16,5))
#plot_history(history,['loss','val_loss'])
#plt.show()
