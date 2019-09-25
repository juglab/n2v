#!/usr/bin/env python
# Noise2Void - 2D Example for RGB Data
from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from csbdeep.io import save_tiff_imagej_compatible

# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
model_name = 'n2v_2D_RGB'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

# In case you do not want to load the weights that lead to lowest validation loss during
# training but the latest computed weights, you can execute the following line:
#model.load_weights('weights_last.h5')

# We read the image we want to process and get rid of the Alpha channel.
img = imread('data/longBeach.png')[...,:3]

# Here we process the image.
pred = model.predict(img, axes='YXC')

from matplotlib.image import imsave
imsave('models/n2v_2D_RGB/pred_longBeach.png', np.clip(pred,0.0,1.0))

# Let's look at the results.
#plt.figure(figsize=(30,30))
# We show the noisy input...
#plt.subplot(1,2,1)
#plt.imshow( img[:,1000:3000,...] )
#plt.title('Input')
# and the result.
#plt.subplot(1,2,2)
#plt.imshow( pred[:,1000:3000,...] )
#plt.title('Prediction')
#plt.show()

# Channel first test
img = np.moveaxis(img, -1, 0)
pred = model.predict(img, axes='CYX')
assert pred.shape == img.shape
