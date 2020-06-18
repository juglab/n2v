#!/usr/bin/env python
# Noise2Void - 2D Example for SEM data
from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible

# A previously trained model is loaded by creating a new N2V-object without providing a 'config'. 
model_name = 'n2v_2D_SEM'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

input_train = imread('data/train.tif')
input_val = imread('data/validation.tif')
pred_train = model.predict(input_train, axes='YX', n_tiles=(2,1))
pred_val = model.predict(input_val[np.newaxis], axes='CYX', n_tiles=(1,2,2))

#plt.figure(figsize=(16,8))
#plt.subplot(1,2,1)
#plt.imshow(input_train[:1500:,:1500],cmap="magma")
#plt.title('Input');
#plt.subplot(1,2,2)
#plt.imshow(pred_train[:1500,:1500],cmap="magma")
#plt.title('Prediction')
#plt.show()

# Let's look at the results
#plt.figure(figsize=(16,8))
#plt.subplot(1,2,1)
#plt.imshow(input_val,cmap="magma")
#plt.title('Input')
#plt.subplot(1,2,2)
#plt.imshow(pred_val,cmap="magma")
#plt.title('Prediction')
#plt.show()

save_tiff_imagej_compatible('models/n2v_2D_SEM/pred_train.tif', pred_train, axes='YX')
save_tiff_imagej_compatible('models/n2v_2D_SEM/pred_validation.tif', pred_val, axes='CYX')
