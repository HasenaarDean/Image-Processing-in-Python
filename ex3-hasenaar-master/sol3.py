############################## Imports ######################################

from skimage.color import rgb2gray
import numpy as np
import os
from imageio import imread
import matplotlib.pyplot as plot
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


def relpath(filename):
    """
    This function reads file by relative path
    :param filename: file name
    :return: file
    """

    return os.path.join(os.path.dirname(__file__), filename)


def enlarge_im(filter_vec, pic):

    """
    This function enlarges the image
    :param filter_vec: filter vector
    :param pic: the picture to enlarge
    :return: the enlarged image
    """

    filter_vec = filter_vec * 2
    filter_vec_trans = filter_vec.transpose()
    rows = pic.shape[0]
    rows = rows * 2
    cols = pic.shape[1]
    cols = cols * 2

    zero_pad = np.zeros((rows, cols))
    zero_pad[::SAMPLING_STEP, ::SAMPLING_STEP] = pic

    first_con = convolve(zero_pad, filter_vec)
    second_con = convolve(first_con, filter_vec_trans)

    return second_con


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


def build_laplacian_pyramid(im, max_levels, filter_size):

    """
    This function calculates the laplacian pyramid.
    :param im: the base image
    :param max_levels: number of max levels.
    :param filter_size: size needed.
    :return: the laplacian pyramid.
    """

    pyr_gau, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    num_of_steps = len(pyr_gau)
    num_of_steps = num_of_steps - 1

    pyr = []

    for level in range(num_of_steps):

        temp_level_pic = pyr_gau[level]
        temp_level_pic = temp_level_pic - enlarge_im(filter_vec, pyr_gau[
            1 + level])
        pyr.append(temp_level_pic)

    pyr.append(pyr_gau[TOP_LAYER])

    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):

    """
    This function calculates the reconstruction of an image from its
    Laplacian Pyramid.
    :param lpyr: Laplacian Pyramid.
    :param filter_vec: filter vector.
    :param coeff: the coefficients.
    :return: the reconstruction of an image from its Laplacian Pyramid.
    """

    new_lap_pyr = np.multiply(lpyr, coeff)
    num_of_steps = len(new_lap_pyr)
    num_of_steps = num_of_steps - 1
    pic = new_lap_pyr[TOP_LAYER]

    for level in range(num_of_steps, 0, -1):
        pic = enlarge_im(filter_vec, pic) + new_lap_pyr[level - 1]

    return pic


def error_msg(msg):

    """
    This function prints error messages.
    :param msg: the message to print.
    :return: error.
    """

    print(msg)


def check_levels_num(levels, max_levels_num):

    """
    This function checks if the number of levels is legal.
    :param levels: number of levels.
    :param max_levels_num: maximum number of levels.
    :return: true or false.
    """

    if levels < MIN_LEVELS_NUM:
        error_msg(MIN_LEVELS_ERROR_MSG)
        return

    if levels > max_levels_num:
        error_msg(MAX_LEVELS_ERROR_MSG)
        return


def calculate_render_levels(pyr, temp_col_num, res, levels):

    """
    This function calculates render levels
    :param pyr: Gaussian or Laplacian pyramid.
    :param temp_col_num: current column.
    :param res: a single black image in which the pyramid levels of the given
    pyramid pyr are stacked horizontally (after stretching the values to
    [0, 1]).
    :param levels: num of levels.
    """

    for i in range(levels):

        temp_level_pic = pyr[i]
        temp_row_num = len(temp_level_pic)
        temp_end_col = temp_row_num + temp_col_num

        temp_level_pic = temp_level_pic - np.nanmin(pyr[i])
        factor_of_normalization = np.nanmax(pyr[i]) - np.nanmin(pyr[i])
        normalized_pic = temp_level_pic / factor_of_normalization

        res[FIRST_ROW:temp_row_num, temp_col_num:temp_end_col] = \
            normalized_pic

        temp_col_num = temp_end_col


def render_pyramid(pyr, levels):

    """
    This function calculates the display of pyramids.
    :param pyr: a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result.
    :return: the render pyramid.
    """

    render_cols = 0
    render_rows = len(pyr[0])

    max_levels_num = len(pyr)
    check_levels_num(levels, max_levels_num)

    for i in range(levels):
        render_cols = len(pyr[i]) + render_cols

    temp_col = 0

    res = np.zeros((render_rows, render_cols)).astype(FLOAT64_TYPE)

    calculate_render_levels(pyr, temp_col, res, levels)

    return res


def display_pyramid(pyr, levels):

    """
    This function displays the pyramid.
    :param pyr: the pyramid to present.
    :param levels: the number of levels to present in the result.
    """

    plot.imshow(render_pyramid(pyr, levels), cmap=CONST_GRAY)
    plot.show()


def calculate_blend_pic(temp_gau_pyr, temp_lap1_pyr, temp_lap2_pyr):

    """
    This function calculates the blended picture.
    :param temp_gau_pyr: Gaussian pyramid
    :param temp_lap1_pyr: Laplacian pyramid 1
    :param temp_lap2_pyr: Laplacian pyramid 2
    :return: blended picture.
    """

    return (temp_lap1_pyr * temp_gau_pyr) + (temp_lap2_pyr * (1 -
                                                              temp_gau_pyr))


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):

    """
    This function calculates the blended picture.
    :param im1: image 1
    :param im2: image 2
    :param mask: mask
    :param max_levels: maximum number of levels.
    :param filter_size_im: filter size of the image
    :param filter_size_mask: filter size of mask
    :return: pyramid blending.
    """

    new_laplacian_pyr = []

    gau_pyr, gau_filter_vec = build_gaussian_pyramid(mask.astype(FLOAT64_TYPE),
                                                     max_levels,
                                                     filter_size_mask)

    lap1_pyr, lap1_filter_vec = build_laplacian_pyramid(im1, max_levels,
                                                        filter_size_im)

    num_of_levels = len(lap1_pyr)

    lap2_pyr, lap2_filter_vec = build_laplacian_pyramid(im2, max_levels,
                                                        filter_size_im)

    for level in range(num_of_levels):

        temp_gau_pyr = gau_pyr[level]
        temp_lap1_pyr = lap1_pyr[level]
        temp_lap2_pyr = lap2_pyr[level]

        temp_level_image = calculate_blend_pic(temp_gau_pyr, temp_lap1_pyr,
                                               temp_lap2_pyr)

        new_laplacian_pyr.append(temp_level_image)

    blended_pic = laplacian_to_image(new_laplacian_pyr, gau_filter_vec,
                                     num_of_levels *
                                     IDENTITY_COEFFICIENTS_LIST)

    blended_pic = np.clip(blended_pic, 0, 1)

    return blended_pic


def blend_channel(channel, pic1, pic2, filter_mask, filter_im, bool_mask,
                  num_of_max_levels):

    """
    This function blends by channel.
    :param channel: R / G / B
    :param pic1: picture 1
    :param pic2: picture 2
    :param filter_mask: filter mask
    :param filter_im: filter image
    :param bool_mask: boolean mask
    :param num_of_max_levels: num of max levels
    :return: blended channel.
    """

    return pyramid_blending(pic1[:, :, channel], pic2[:, :, channel],
                            bool_mask, num_of_max_levels, filter_im,
                            filter_mask)


def generate_blended_rgb_image(pic1, pic2, filter_mask, filter_im,
                               bool_mask, num_of_max_levels):

    """
    This function generates blended rgb image.
    :param pic1: picture 1
    :param pic2: picture 2
    :param filter_mask: filter mask
    :param filter_im: filter image
    :param bool_mask: boolean mask
    :param num_of_max_levels: num of max levels
    :return: blended rgb image.
    """

    pic = np.zeros(pic1.shape)

    for c in range(NUM_OF_CHANNELS_RGB):
        pic[:, :, c] = blend_channel(c, pic1, pic2, filter_mask, filter_im,
                                     bool_mask, num_of_max_levels)

    return pic


def helper_blending_example(mask_path, pic1_path, pic2_path):

    """
    This is a helper function for the functions: blending_example1(),
    blending_example2().

    :param mask_path: mask path
    :param pic1_path: path of picture 1
    :param pic2_path: path of picture 2
    :return: a list of picture 1, picture 2, boolean mask, and the blended
    image.
    """

    bool_mask = read_image(relpath(mask_path),
                           GRAYSCALE_REPRESENTATION).astype(BOOL_TYPE)
    pic1 = read_image(relpath(pic1_path), RGB_REPRESENTATION)
    pic2 = read_image(relpath(pic2_path), RGB_REPRESENTATION)
    res = generate_blended_rgb_image(pic1, pic2, 3, 28, bool_mask, 10)

    plot.figure()
    plot.subplot(2, 2, 1)
    plot.imshow(pic1)
    plot.subplot(2, 2, 2)
    plot.imshow(pic2)
    plot.subplot(2, 2, 3)
    plot.imshow(bool_mask, cmap=CONST_GRAY)
    plot.subplot(2, 2, 4)
    plot.imshow(res)
    plot.show()
    answer = [pic1, pic2, bool_mask, res]
    return answer


def blending_example1():

    """
    This function prints out the first blending example.
    :return: a list of picture 1, picture 2, boolean mask, and the blended
    image.
    """

    return helper_blending_example(EXAMPLE_1_MASK_PATH,
                                   EXAMPLE_1_PIC_1_PATH, EXAMPLE_1_PIC_2_PATH)


def blending_example2():

    """
    This function prints out the second blending example.
    :return: a list of picture 1, picture 2, boolean mask, and the blended
    image.
    """

    return helper_blending_example(EXAMPLE_2_MASK_PATH,
                                   EXAMPLE_2_PIC_1_PATH, EXAMPLE_2_PIC_2_PATH)
