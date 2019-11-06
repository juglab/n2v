#!/usr/bin/env python3

import os
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="base directory in which your network will live", default='models')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your training data")
parser.add_argument("--fileName", help="name of your training data file", default="*.tif")
parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent)", default=5.0, type=float)
parser.add_argument("--dims", help="dimensions of your data, can include: X,Y,Z,C (channel), T (time)", default='YX')
parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=64, type=int)
parser.add_argument("--patchSizeZ", help="Z-size of your training patches", default=64, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=100, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=400, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=64, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=2, type=int)
parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V", default=1.6, type=float)
parser.add_argument("--learningRate", help="initial learning rate", default=0.0004, type=float)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=32, type=int)
parser.add_argument("--noAugment",  action='store_true', help="do not rotate and flip training patches")

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)

from n2v.models import N2VConfig, N2V
print('everything imported')
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile

from tifffile import imread
from tifffile import imwrite


import glob
print('everything imported')


print("args",str(args.name))

print('augment',(not args.noAugment))



####################################################
#           PREPARE TRAINING DATA
####################################################


datagen = N2V_DataGenerator()
imgs = datagen.load_imgs_from_directory(directory = args.dataPath, dims=args.dims, filter=args.fileName)
print("imgs.shape",imgs[0].shape)

# Here we extract patches for training and validation.
pshape=( args.patchSizeXY, args.patchSizeXY)
if 'Z' in args.dims:
    pshape=(args.patchSizeZ, args.patchSizeXY, args.patchSizeXY)

print(pshape)
patches = datagen.generate_patches_from_list(imgs[:1], shape=pshape, augment=(not args.noAugment))

# The patches are non-overlapping, so we can split them into train and validation data.
frac= int( (len(patches))*float(args.validationFraction)/100.0)
print("total no. of patches: "+str(len(patches)) + "\ttraining patches: "+str(len(patches)-frac)+"\tvalidation patches: "+str(frac))
X = patches[frac:]
X_val = patches[:frac]



config = N2VConfig(X, unet_kern_size=args.netKernelSize,
                   train_steps_per_epoch=int(args.stepsPerEpoch),train_epochs=int(args.epochs), train_loss='mse', batch_norm=True,
                   train_batch_size=args.batchSize, n2v_perc_pix=args.n2vPercPix, n2v_patch_shape=pshape,
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_learning_rate=args.learningRate,
                   unet_n_depth=args.netDepth,
                   unet_n_first=args.unet_n_first
                   )

# Let's look at the parameters stored in the config-object.
vars(config)


# a name used to identify the model
model_name = args.name
# the base directory in which our model will live
basedir = args.baseDir
# We are now creating our network model.
model = N2V(config=config, name=model_name, basedir=basedir)



####################################################
#           Train Network
####################################################
print("begin training")
history = model.train(X, X_val)

