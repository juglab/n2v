from pathlib import Path
import tifffile

"""
 Original non compressed file can be downloaded from https://imagej.nih.gov/ij/images/flybrain.zip
 LZW compressed file obtained from https://online-converting.com/image/convert2tiff
"""


def test_tifffile():
    current = Path(__file__)
    image_lzw = Path(current.parent, 'test_data/flybrain_lzw.tiff')

    image = tifffile.imread(image_lzw)

    assert image.shape == (256, 256, 3)
