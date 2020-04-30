import numpy as np
from os.path import join
from glob import glob
import tifffile
from matplotlib import image
from csbdeep.utils import _raise

class N2V_DataGenerator():
    """
    The 'N2V_DataGenerator' enables training and validation data generation for Noise2Void.
    """

    def load_imgs(self, files, dims='YX'):
        """
        Helper to read a list of files. The images are not required to have same size,
        but have to be of same dimensionality.

        Parameters
        ----------
        files  : list(String)
                 List of paths to tiff-files.
        dims   : String, optional(default='YX')
                 Dimensions of the images to read. Known dimensions are: 'TZYXC'

        Returns
        -------
        images : list(array(float))
                 A list of the read tif-files. The images have dimensionality 'SZYXC' or 'SYXC'
        """
        assert 'Y' in dims and 'X' in dims, "'dims' has to contain 'X' and 'Y'."

        tmp_dims = dims
        for b in ['X', 'Y', 'Z', 'T', 'C']:
            assert tmp_dims.count(b) <= 1, "'dims' has to contain {} at most once.".format(b)
            tmp_dims = tmp_dims.replace(b, '')

        assert len(tmp_dims) == 0, "Unknown dimensions in 'dims'."

        if 'Z' in dims:
            net_axes = 'ZYXC'
        else:
            net_axes = 'YXC'

        move_axis_from = ()
        move_axis_to = ()
        for d, b in enumerate(dims):
            move_axis_from += tuple([d])
            if b == 'T':
                move_axis_to += tuple([0])
            elif b == 'C':
                move_axis_to += tuple([-1])
            elif b in 'XYZ':
                if 'T' in dims:
                    move_axis_to += tuple([net_axes.index(b)+1])
                else:
                    move_axis_to += tuple([net_axes.index(b)])
        imgs = []
        for f in files:
            if f.endswith('.tif') or f.endswith('.tiff'):
                imread = tifffile.imread
            elif f.endswith('.png'):
                imread = image.imread
            elif f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.JPG'):
                _raise(Exception("JPEG is not supported, because it is not loss-less and breaks the pixel-wise independence assumption."))
            else:
                _raise("Filetype '{}' is not supported.".format(f))

            img = imread(f).astype(np.float32)
            assert len(img.shape) == len(dims), "Number of image dimensions doesn't match 'dims'."

            img = np.moveaxis(img, move_axis_from, move_axis_to)

            if not ('T' in dims):    
                img = img[np.newaxis]

            if not ('C' in dims):
                img = img[..., np.newaxis]

            imgs.append(img)

        return imgs

    def load_imgs_from_directory(self, directory, filter='*.tif', dims='YX'):
        """
        Helper to read all files which match 'filter' from a directory. The images are not required to have same size,
        but have to be of same dimensionality.

        Parameters
        ----------
        directory : String
                    Directory from which the data is loaded.
        filter    : String, optional(default='*.tif')
                    Filter to match the file names.
        dims      : String, optional(default='YX')
                    Dimensions of the images to read. Known dimensions are: 'TZYXC'

        Returns
        -------
        images : list(array(float))
                 A list of the read tif-files. The images have dimensionality 'SZYXC' or 'SYXC'
        """

        files = glob(join(directory, filter))
        files.sort()
        return self.load_imgs(files, dims=dims)


    def generate_patches_from_list(self, data, num_patches_per_img=None, shape=(256, 256), augment=True, shuffle=False):
        """
        Extracts patches from 'list_data', which is a list of images, and returns them in a 'numpy-array'. The images
        can have different dimensionality.

        Parameters
        ----------
        data                : list(array(float))
                              List of images with dimensions 'SZYXC' or 'SYXC'
        num_patches_per_img : int, optional(default=None)
                              Number of patches to extract per image. If 'None', as many patches as fit i nto the
                              dimensions are extracted.
        shape               : tuple(int), optional(default=(256, 256))
                              Shape of the extracted patches.
        augment             : bool, optional(default=True)
                              Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
        shuffle             : bool, optional(default=False)
                              Shuffles extracted patches across all given images (data).

        Returns
        -------
        patches : array(float)
                  Numpy-Array with the patches. The dimensions are 'SZYXC' or 'SYXC'
        """
        patches = []
        for img in data:
            for s in range(img.shape[0]):
                p = self.generate_patches(img[s][np.newaxis], num_patches=num_patches_per_img, shape=shape, augment=augment)
                patches.append(p)

        patches = np.concatenate(patches, axis=0)

        if shuffle:
            np.random.shuffle(patches)

        return patches

    def generate_patches(self, data, num_patches=None, shape=(256, 256), augment=True):
        """
        Extracts patches from 'data'. The patches can be augmented, which means they get rotated three times
        in XY-Plane and flipped along the X-Axis. Augmentation leads to an eight-fold increase in training data.

        Parameters
        ----------
        data        : list(array(float))
                      List of images with dimensions 'SZYXC' or 'SYXC'
        num_patches : int, optional(default=None)
                      Number of patches to extract per image. If 'None', as many patches as fit i nto the
                      dimensions are extracted.
        shape       : tuple(int), optional(default=(256, 256))
                      Shape of the extracted patches.
        augment     : bool, optional(default=True)
                      Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.

        Returns
        -------
        patches : array(float)
                  Numpy-Array containing all patches (randomly shuffled along S-dimension).
                  The dimensions are 'SZYXC' or 'SYXC'
        """

        patches = self.__extract_patches__(data, num_patches=num_patches, shape=shape, n_dims=len(data.shape)-2)
        if shape[-2] == shape[-1]:
            if augment:
                patches = self.__augment_patches__(patches=patches)
        else:
            if augment:
                print("XY-Plane is not square. Omit augmentation!")

        np.random.shuffle(patches)
        print('Generated patches:', patches.shape)
        return patches

    def __extract_patches__(self, data, num_patches=None, shape=(256, 256), n_dims=2):
        if num_patches == None:
            patches = []
            if n_dims == 2:
                if data.shape[1] > shape[0] and data.shape[2] > shape[1]:
                    for y in range(0, data.shape[1] - shape[0] + 1, shape[0]):
                        for x in range(0, data.shape[2] - shape[1] + 1, shape[1]):
                            patches.append(data[:, y:y + shape[0], x:x + shape[1]])

                    return np.concatenate(patches)
                elif data.shape[1] == shape[0] and data.shape[2] == shape[1]:
                    return data
                else:
                    print("'shape' is too big.")
            elif n_dims == 3:
                if data.shape[1] > shape[0] and data.shape[2] > shape[1] and data.shape[3] > shape[2]:
                    for z in range(0, data.shape[1] - shape[0] + 1,  shape[0]):
                        for y in range(0, data.shape[2] - shape[1] + 1, shape[1]):
                            for x in range(0, data.shape[3] - shape[2] + 1, shape[2]):
                                patches.append(data[:, z:z + shape[0], y:y + shape[1], x:x + shape[2]])

                    return np.concatenate(patches)
                elif data.shape[1] == shape[0] and data.shape[2] == shape[1] and data.shape[3] == shape[2]:
                    return data
                else:
                    print("'shape' is too big.")
            else:
                print('Not implemented for more than 4 dimensional (ZYXC) data.')
        else:
            patches = []
            if n_dims == 2:
                for i in range(num_patches):
                    y, x = np.random.randint(0, data.shape[1] - shape[0] + 1), np.random.randint(0,
                                                                                                 data.shape[
                                                                                                          2] - shape[
                                                                                                          1] + 1)
                    patches.append(data[0, y:y + shape[0], x:x + shape[1]])

                if len(patches) > 1:
                    return np.stack(patches)
                else:
                    return np.array(patches)[np.newaxis]
            elif n_dims == 3:
                for i in range(num_patches):
                    z, y, x = np.random.randint(0, data.shape[1] - shape[0] + 1), np.random.randint(0,
                                                                                                    data.shape[
                                                                                                             2] - shape[
                                                                                                             1] + 1), np.random.randint(
                        0, data.shape[3] - shape[2] + 1)
                    patches.append(data[0, z:z + shape[0], y:y + shape[1], x:x + shape[2]])

                if len(patches) > 1:
                    return np.stack(patches)
                else:
                    return np.array(patches)[np.newaxis]
            else:
                print('Not implemented for more than 4 dimensional (ZYXC) data.')

    def __augment_patches__(self, patches):
        if len(patches.shape[1:-1]) == 2:
            augmented = np.concatenate((patches,
                                        np.rot90(patches, k=1, axes=(1, 2)),
                                        np.rot90(patches, k=2, axes=(1, 2)),
                                        np.rot90(patches, k=3, axes=(1, 2))))
        elif len(patches.shape[1:-1]) == 3:
            augmented = np.concatenate((patches,
                                        np.rot90(patches, k=1, axes=(2, 3)),
                                        np.rot90(patches, k=2, axes=(2, 3)),
                                        np.rot90(patches, k=3, axes=(2, 3))))

        augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
        return augmented
