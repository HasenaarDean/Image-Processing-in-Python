############################## Imports ######################################

from scipy.signal import convolve2d
import numpy as np

from skimage.color import rgb2gray
from imageio import imread
from scipy.ndimage.filters import convolve

########################## Global constants  ################################

FACTOR_OF_NORMALIZATION = 255
LEN_OF_GRAYSCALE_SHAPE = 2
RGB_REPRESENTATION = 2
GRAYSCALE_REPRESENTATION = 1
FLOAT64_TYPE = np.float64
BOOL_TYPE = np.bool
NUM_OF_CHANNELS_RGB = 3
BASE_OF_PASCALS_TRIANGLE = np.array([1, 1])
SHAPE_OF_FILTER_VEC = (1, -1)
MINIMUM_DIMENSION = 16
CONST_GRAY = "gray"
FIRST_ROW = 0
IDENTITY_COEFFICIENTS_LIST = [1]
TOP_LAYER = -1
MIN_LEVELS_NUM = 1
SAMPLING_STEP = 2
MIN_LEVELS_ERROR_MSG = "Number of levels must be at least 1"
MAX_LEVELS_ERROR_MSG = "Number of levels must not be bigger than the " \
                       "pyramid's number of levels"

EXAMPLE_1_MASK_PATH = "externals/TLV_MASK.jpg"
EXAMPLE_1_PIC_1_PATH = "externals/UFO.jpg"
EXAMPLE_1_PIC_2_PATH = "externals/TLV.jpg"

EXAMPLE_2_MASK_PATH = "externals/WINDOW_MASK.jpg"
EXAMPLE_2_PIC_1_PATH = "externals/JOKER.jpg"
EXAMPLE_2_PIC_2_PATH = "externals/WINDOW.jpg"

############################# Functions #####################################


def read_image(filename, representation):

    """
    This function reads an image file and converts it into a given
    representation.
    :param filename: the name of the file.
    :param representation: grayscale (1) or RGB (2).
    :return: The processed image matrix.
    """

    pic = imread(filename).astype(np.float64)
    normalized_pic = pic / FACTOR_OF_NORMALIZATION
    pic_dimension = len(pic.shape)

    if representation != RGB_REPRESENTATION and representation != \
            GRAYSCALE_REPRESENTATION:
        return

    if representation == RGB_REPRESENTATION or \
            pic_dimension == LEN_OF_GRAYSCALE_SHAPE:
        return normalized_pic
    else:
        return rgb2gray(normalized_pic)


def make_filter_vec(size_of_filter):

    """
    This function calculates the filter vector.
    :param size_of_filter: size needed.
    :return: filter vector.
    """

    filter_vec = BASE_OF_PASCALS_TRIANGLE
    num_of_convolutions = size_of_filter - 2

    for con in range(num_of_convolutions):
        filter_vec = np.convolve(filter_vec, BASE_OF_PASCALS_TRIANGLE)

    temp_sum = filter_vec.sum()
    filter_vec = filter_vec.reshape(SHAPE_OF_FILTER_VEC)
    filter_vec = filter_vec / temp_sum

    return filter_vec


def bigger_than_minimum_dimension(rows, cols):

    """
    This function checks dimension of an image.
    :param rows: number of rows
    :param cols: number of columns.
    :return: true or false.
    """

    return cols >= MINIMUM_DIMENSION and rows >= MINIMUM_DIMENSION


def build_gaussian_pyramid(im, max_levels, filter_size):

    """
    This function calculates the gaussian pyramid.
    :param im: the base image
    :param max_levels: number of max levels.
    :param filter_size: size needed.
    :return: the gaussian pyramid.
    """

    pyr = []
    filter_vec = make_filter_vec(filter_size)
    filter_vec_trans = filter_vec.transpose()

    temp_im = im.copy()
    pyr.append(im)

    for i in range(1, max_levels):
        temp_im = convolve(temp_im, filter_vec)
        temp_im = convolve(temp_im, filter_vec_trans)
        temp_im = temp_im[::SAMPLING_STEP, ::SAMPLING_STEP]
        rows = temp_im.shape[0]
        cols = temp_im.shape[1]

        if not bigger_than_minimum_dimension(rows, cols):
            break

        pyr.append(temp_im)

    return pyr, filter_vec


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img