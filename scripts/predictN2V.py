import os
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="directory in which all your network will live", default='models')
parser.add_argument("--name", help="name of your network", default='N2V3D')
parser.add_argument("--testData", help="The path to your test data")
parser.add_argument("--output", help="The path to your data to be saved", default='predictions.tif')
parser.add_argument("--dims", help="dimensions of your data", default='YX')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

# We import all our dependencies.
from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from tifffile import imwrite


# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
model_name = 'n2v_3D'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)


# Load the test data.
img = imread(args.testData)

# Denoise the image.
pred = model.predict( img, axes=args.dims )
print(pred.shape)

imwrite(args.output,pred.astype(np.float32))
