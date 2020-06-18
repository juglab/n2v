import pytest
import tifffile
from PIL import Image

'''
 Original non compressed file can be downloaded from https://imagej.nih.gov/ij/images/flybrain.zip
 LZW compressed file obtained from https://online-converting.com/image/convert2tiff
'''
class TestLZWCompressedTiff():
 
    def test_tifffile(self):
        image_lzw = 'flybrain_lzw.tiff'
        tifffile.imread(image_lzw)
        
    def test_pillow(self):
        image_lzw = 'flybrain_lzw.tiff'
        im1 = Image.open(image_lzw)