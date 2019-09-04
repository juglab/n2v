#!/usr/bin/env python
# Noise2Void - 3D Example for Flywing Data"
from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible

model_name = 'n2v_3D'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

img = imread('data/flywing.tif')
pred = model.predict(img, axes='ZYX', n_tiles=(2,4,4))

#plt.figure(figsize=(30,30))
#plt.subplot(1,2,1)
#plt.imshow(np.max(img,axis=0),
#           cmap='magma',
#           vmin=np.percentile(img,0.1),
#           vmax=np.percentile(img,99.9))
#plt.title('Input')
#plt.subplot(1,2,2)\n",
#plt.imshow(np.max(pred,axis=0),
#    cmap='magma',
#    vmin=np.percentile(img,0.1),
#    vmax=np.percentile(img,99.9))
#plt.title('Prediction')

save_tiff_imagej_compatible('models/n2v_3D/prediction.tif', pred, 'ZYX')
