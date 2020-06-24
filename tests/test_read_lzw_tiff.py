import pytest
from _pytest.outcomes import fail
import tifffile

'''
 Original non compressed file can be downloaded from https://imagej.nih.gov/ij/images/flybrain.zip
 LZW compressed file obtained from https://online-converting.com/image/convert2tiff
'''
class TestLZWCompressedTiff():
 
    def test_tifffile(self):
        image_lzw = 'test_data/flybrain_lzw.tiff'
        try:
            tifffile.imread(image_lzw)
        except Exception as e:
            fail(msg='Unable to read LZW compressed TIFF')
        