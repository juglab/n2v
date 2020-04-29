#!/usr/bin/env python
# Noise2Void - 2D Example for SEM data
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile

# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('./data')
zipPath = "data/SEM.zip"
if not os.path.exists(zipPath):
    # download and unzip data
    data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/pXgfbobntrw06lC/download', zipPath)
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall("data")

# We create our DataGenerator-object.
# It will help us load data and extract patches for training and validation.
datagen = N2V_DataGenerator()

# We load all the '.tif' files from the 'data' directory.
# The function will return a list of images (numpy arrays
imgs = datagen.load_imgs_from_directory(directory = "data/")
   
# Let's look at the shape of the images.
print(imgs[0].shape,imgs[1].shape)
# The function automatically added two extra dimensions to the images:
# One at the beginning, is used to hold a potential stack of images such as a movie.
# One at the end, represents channels.
# Lets' look at the images
# We have to remove the added extra dimensions to display them as 2D images.
#plt.imshow(imgs[0][0,...,0], cmap='magma')
#plt.show()
#plt.imshow(imgs[1][0,...,0], cmap='magma')
#plt.show()"

# We will use the first image to extract training patches and store them in 'X'
X = datagen.generate_patches_from_list(imgs[:1], shape=(96,96))
# We will use the second image to extract validation patches
X_val = datagen.generate_patches_from_list(imgs[1:], shape=(96,96))

# Let's look at one of our training and validation patches.
#plt.figure(figsize=(14,7))
#plt.subplot(1,2,1)
#plt.imshow(X[0,...,0], cmap='magma')
#plt.title('Training Patch')
#plt.subplot(1,2,2)
#plt.imshow(X_val[0,...,0], cmap='magma')
#plt.title('Validation Patch')
#plt.show()

# You can increase "train_steps_per_epoch" to get even better results at the price of longer computation.
config = N2VConfig(X, unet_kern_size=3,
                   train_steps_per_epoch=10,train_epochs=10, train_loss='mse', batch_norm=True,
                   train_batch_size=128, n2v_perc_pix=1.6, n2v_patch_shape=(64, 64),
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)
# Let's look at the parameters stored in the config-object.
vars(config)

# a name used to identify the model
model_name = 'n2v_2D_SEM'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model.
model = N2V(config, model_name, basedir=basedir)

# We are ready to start training now.
history = model.train(X, X_val)
print(sorted(list(history.history.keys())))
#plt.figure(figsize=(16,5))
#plot_history(history,['loss','val_loss'])
#plt.show()