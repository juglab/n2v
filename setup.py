from __future__ import absolute_import
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'n2v','version.py')) as f:
    exec(f.read())

with open(path.join(_dir,'README.md')) as f:
    long_description = f.read()


setup(name='n2v',
      version=__version__,
      description='Noise2Void allows the training of a denoising CNN from individual noisy images. This implementation'
                  'extends CSBDeep.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/juglab/n2v/',
      author='Tim-Oliver Buchholz, Alexander Krull',
      author_email='tibuch@mpi-cbg.de, krull@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),

      project_urls={
          'Repository': 'https://github.com/juglab/n2v/',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],

      scripts=['scripts/trainN2V.py',
          'scripts/predictN2V.py'
      ],

      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "six",
          "keras>=2.1.1,<2.4.0",
          "tifffile>=2020.5.11",
          "imagecodecs>=2020.2.18",
          "tqdm",
          "backports.tempfile;python_version<'3.4'",
          "csbdeep>=0.6.0,<0.7.0",
          "Pillow",
          "ruamel.yaml>=0.16.10"
        ]
      )
