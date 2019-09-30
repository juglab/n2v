#!/usr/bin/env python
# Noise2Void - 3D Example
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile
# create a folder for data, if it already exists, remove current contents (results of 2D tests)
if not os.path.isdir('./data'):
    os.mkdir('./data')
else:
    for the_file in os.listdir('./data'):
        file_path = os.path.join('./data', the_file)
        os.unlink(file_path)
    
# check if data has been downloaded already
zipPath='data/flywing-data.zip'
if not os.path.exists(zipPath):
    #download and unzip data
    data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/RKStdwKo4FlFrxE/download', zipPath) 
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall('data')
    
# Training Data Preparation
# For training we will load __one__ low-SNR 3D-tif-volume and use the <code>N2V_DataGenerator</code> to extract non-overlapping 3D-patches. Since N2V is a self-supervised method, we don't need targets."
datagen = N2V_DataGenerator()

# We will load all the '.tif' files from the 'data' directory. In our case it is only one.
# The function will return a list of images (numpy arrays).
# In the 'dims' parameter we specify the order of dimensions in the image files we are reading.
imgs = datagen.load_imgs_from_directory(directory = "data/", dims='ZYX')
print(imgs[0].shape)
# The function automatically added two extra dimension to the images:
# One at the front is used to hold a potential stack of images such as a movie.
# One at the end could hold color channels such as RGB.
# Let's look at a maximum projection of the volume.\n",
# We have to remove the added extra dimensions to display it.\n",
#plt.figure(figsize=(32,16))
#plt.imshow(np.max(imgs[0][0,...,0],axis=0),
#    cmap='magma',
#    vmin=np.percentile(imgs[0],0.1),
#    vmax=np.percentile(imgs[0],99.9))
#plt.show()

# Here we extract patches for training and validation.
patches = datagen.generate_patches_from_list(imgs[:1], shape=(32, 64, 64))

# Patches are created so they do not overlap.
# (Note: this is not the case if you specify a number of patches. See the docstring for details!)
# Non-overlapping patches enable us to split them into a training and validation set.
X = patches[:600]
X_val = patches[600:]

# Let's look at two patches
#plt.figure(figsize=(14,7))
#plt.subplot(1,2,1)
#plt.imshow(X[0,16,...,0],cmap='magma')
#plt.title('Training Patch')
#plt.subplot(1,2,2)
#plt.imshow(X_val[0,16,...,0],cmap='magma')
#plt.title('Validation Patch')
#plt.show()

# You can increase "train_steps_per_epoch" to get even better results at the price of longer computation.
config = N2VConfig(X, unet_kern_size=3,
    train_steps_per_epoch=100, train_epochs=10, train_loss='mse', batch_norm=True,
    train_batch_size=4, n2v_perc_pix=1.6, n2v_patch_shape=(32, 64, 64),
    n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)
# Let's look at the parameters stored in the config-object.
vars(config)

# a name used to identify the model
model_name = 'n2v_3D'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model
model = N2V(config=config, name=model_name, basedir=basedir)

history = model.train(X, X_val)
print(sorted(list(history.history.keys())))
#plt.figure(figsize=(16,5))
#plot_history(history,['loss','val_loss'])
#plt.show()
