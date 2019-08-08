#!/usr/bin/env python3

import os
import sys
import argparse
from glob import glob
import csbdeep.io

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="directory in which all your network will live", default='models')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--fileName", help="name of your data file", default="*.tif")
parser.add_argument("--output", help="The path to which your data is to be saved", default='.')
parser.add_argument("--dims", help="dimensions of your data", default='YX')
parser.add_argument("--tile", help="will cut your image [TILE] times in every dimension to make it fit GPU memory", default=1, type=int)

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

print(args.output)

assert (not 'T' in args.dims) or (args.dims[0]=='T')

# We import all our dependencies.
from n2v.models import N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread


# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
model_name = args.name
basedir = args.baseDir
model = N2V(config=None, name=model_name, basedir=basedir)


tiles = (args.tile, args.tile)

if 'Z' in args.dims or 'C' in args.dims:
    tiles = (1, args.tile, args.tile)

if 'Z' in args.dims and 'C' in args.dims:
    tiles = (1, args.tile, args.tile, 1)

datagen = N2V_DataGenerator()
imgs = datagen.load_imgs_from_directory(directory = args.dataPath, dims=args.dims, filter=args.fileName)


files = glob(os.path.join(args.dataPath, args.fileName))
files.sort()

for i, img in enumerate(imgs):
    img_=img

    if 'Z' in args.dims:
        myDims='TZYXC'
    else:
        myDims='TYXC'

    if not 'C' in args.dims :
        img_=img[...,0]
        myDims=myDims[:-1]

    myDims_=myDims[1:]


    if not 'C' in args.dims :
        img_=img[...,0]

    # if we have a time dimension we process the images one by one
    if args.dims[0]=='T':
        outDims=myDims
        pred=img_.copy()
        for j in range(img_.shape[0]):
            print('predicting slice', j, img_[j].shape, myDims_)
            pred[j] = model.predict( img_[j], axes=myDims_, n_tiles=tiles)
    else:
        outDims=myDims_
        img_=img_[0,...]
        print("denoising image "+str(i+1) +" of "+str(len(imgs)))
        # Denoise the image.
        pred = model.predict( img_, axes=myDims_, n_tiles=tiles)

    print(pred.shape)
    outpath=args.output
    filename=os.path.basename(files[i]).replace('.tif','_N2V.tif')
    outpath=os.path.join(outpath,filename)
    print('writing file to ',outpath, outDims, pred.shape)
    csbdeep.io.save_tiff_imagej_compatible(outpath, pred.astype(np.float32), outDims)
