import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve

# tiny error used for nummerical stability
eps = 0.1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    """[summary]
    L^2 norm of a list
    [description]
    Used for internals
    Arguments:
        lst {list} -- vector
    """
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]

    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle

    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)


class RandomInitKernel(object):
    """[summary]

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: tuple = (100, 100)):

        # checking if size is correctly given
        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        # saving args
        self.SIZE = size

        # flag to see if kernel has been calculated already
        self.kernel_is_generated = False


    def _createKernel(self, save_to: Path=None, show: bool=False):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        use displayKernel to actually see the kernel.

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # check if we haven't already generated a kernel
        if self.kernel_is_generated:
            return None

        # (pillow Image object)
        self.kernel_image = Image.new("RGB", self.SIZE)

        # convert to gray scale
        self.kernel_image = self.kernel_image.convert("L")

        # flag that we have generated a kernel
        self.kernel_is_generated = True

    def displayKernel(self, save_to: Path=None, show: bool=True):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        Saves the kernel to save_to if needed or shows it
        is show true

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # generate kernel if needed
        self._createKernel()

        # save if needed
        if save_to is not None:

            save_to_file = Path(save_to)

            # save Kernel image
            self.kernel_image.save(save_to_file)
        else:
            # Show kernel
            self.kernel_image.show()

    @property
    def kernelMatrix(self) -> np.ndarray:
        """[summary]
        Kernel matrix of motion blur of given intensity.
        [description]
        Once generated, it stays the same.
        Returns:
            numpy ndarray
        """

        # generate kernel if needed
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)

        return kernel

    @kernelMatrix.setter
    def kernelMatrix(self, *kargs):
        raise NotImplementedError("Can't manually set kernel matrix yet")

    def applyTo(self, image, keep_image_dim: bool = False) -> Image:
        """[summary]
        Applies kernel to one of the following:

        1. Path to image file
        2. Pillow image object
        3. (H,W,3)-shaped numpy array
        [description]

        Arguments:
            image {[str, Path, Image, np.ndarray]}
            keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

        Returns:
            Image -- [description]
        """
        # calculate kernel if haven't already
        self._createKernel()

        def applyToPIL(image: Image, keep_image_dim: bool = False) -> Image:
            """[summary]
            Applies the kernel to an PIL.Image instance
            [description]
            converts to RGB and applies the kernel to each
            band before recombining them.
            Arguments:
                image {Image} -- Image to convolve
                keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

            Returns:
                Image -- blurred image
            """
            # convert to RGB
            image = image.convert(mode="RGB")

            conv_mode = "valid"
            if keep_image_dim:
                conv_mode = "same"

            result_bands = ()

            for band in image.split():

                # convolve each band individually with kernel
                result_band = convolve(
                    band, self.kernelMatrix, mode=conv_mode).astype("uint8")

                # collect bands
                result_bands += result_band,

            # stack bands back together
            result = np.dstack(result_bands)

            # Get image
            return Image.fromarray(result)

        # If image is Path
        if isinstance(image, str) or isinstance(image, Path):

            # open image as Image class
            image_path = Path(image)
            image = Image.open(image_path)

            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, Image.Image):

            # apply kernel
            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, np.ndarray):

            # ASSUMES we have an array of the form (H, W, 3)
            ###

            # initiate Image object from array
            image = Image.fromarray(image)

            return applyToPIL(image, keep_image_dim)

        else:

            raise ValueError("Cannot apply kernel to this type.")


if __name__ == '__main__':
    image = Image.open("./images/moon.png")
    image.show()
    k = RandomInitKernel()

    k.applyTo(image, keep_image_dim=True).show()