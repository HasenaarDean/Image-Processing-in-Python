############################## Imports ######################################

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from imageio import imread

########################## Global constants #################################

INT8 = np.uint8
FLOAT64 = np.float64
LOWER_INTERPOLATION = "lower"
COORDINATE_OF_Y = 0
FACTOR_OF_NORMALIZATION = 255
VALUE_OF_FIRST_Z = 0
VALUE_OF_LAST_Z = 255
MAX_GRAY_VAL = 255
MIN_GRAY_VAL = 0
LEN_OF_RGB_SHAPE = 3
LEN_OF_GRAYSCALE_SHAPE = 2
RGB_REPRESENTATION = 2
GRAYSCALE_REPRESENTATION = 1
NUM_OF_BINS = 256
RANGE_OF_BINS = [0, 1]
LAST_ELEMENT_INDEX = -1
ROWS_COORDS = 0
COLS_COORDS = 1
DEPTH_COORDS = 2

RGB_TO_YIQ_MATRIX_TRANS = np.array(
    [[0.299, 0.596, 0.212],
     [0.587, -0.275, -0.523],
     [0.114, -0.321, 0.311]])

YIQ_TO_RGB_MATRIX_TRANS = np.linalg.inv(RGB_TO_YIQ_MATRIX_TRANS)


############################# Functions ####################################


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


def imdisplay(filename, representation):

    """
    This function displays an image in a given representation.
    :param filename: the name of the file.
    :param representation: representation: grayscale (1) or RGB (2).
    """

    pic = read_image(filename, representation)
    if representation == GRAYSCALE_REPRESENTATION:
        plt.imshow(pic, cmap=plt.cm.gray)
    else:
        plt.imshow(pic)

    plt.show()


def rgb2yiq(imRGB):

    """
    This function transforms an RGB image into the YIQ color space.
    :param imRGB: RGB image input.
    :return: YIQ image output.
    """

    return np.dot(imRGB[:, :, :LEN_OF_RGB_SHAPE], RGB_TO_YIQ_MATRIX_TRANS)


def yiq2rgb(imYIQ):

    """
    This function transforms an YIQ image into the RGB color space.
    :param imYIQ: YIQ image input.
    :return: RGB image output.
    """

    return np.dot(imYIQ[:, :, :LEN_OF_RGB_SHAPE], YIQ_TO_RGB_MATRIX_TRANS)


def histogram_equalize(im_orig):

    """
    This function performs histogram equalization of a given grayscale or
    RGB image.
    :param im_orig: the input image.
    :return: histogram equalization of the input image.
    """

    is_rgb = len(im_orig.shape) == LEN_OF_RGB_SHAPE
    if is_rgb:
        temp_yiq = rgb2yiq(im_orig)
        target_pic = temp_yiq[:, :, COORDINATE_OF_Y]
    else:
        target_pic = im_orig

    # Compute the image histogram:
    hist_orig, edge_of_bins = np.histogram(target_pic, NUM_OF_BINS,
                                           RANGE_OF_BINS)

    # Compute the cumulative histogram:
    cumulative_hist = np.cumsum(hist_orig)

    # Normalize the cumulative histogram (divide by the total number of
    # pixels):
    pix_num = cumulative_hist[MAX_GRAY_VAL]
    cumulative_hist_normalized = cumulative_hist / pix_num

    # Multiply the normalized histogram by the maximal gray level value:
    cumulative_hist_normalized = cumulative_hist_normalized * MAX_GRAY_VAL

    # Verify that the minimal value is 0 and that the maximal is 255:

    cum_min = cumulative_hist_normalized.min()
    cum_max = cumulative_hist_normalized.max()
    diff = cum_max - cum_min

    stretch_verifier = (cum_min == MIN_GRAY_VAL) and \
                       (cum_max == MAX_GRAY_VAL)

    # Otherwise, stretch the result linearly in the range:
    if not stretch_verifier:
        cumulative_hist_normalized = (cumulative_hist_normalized - cum_min) \
                                     * (MAX_GRAY_VAL / diff)

    # Round the values to get integers:
    lookup_table = np.round(cumulative_hist_normalized)

    if is_rgb:
        temp_yiq[:, :, COORDINATE_OF_Y] = \
            lookup_table.flat[(target_pic * MAX_GRAY_VAL).astype(int)] \
            / MAX_GRAY_VAL
        im_eq = yiq2rgb(temp_yiq)
        hist_eq, edge_of_bins = np.histogram(temp_yiq[:, :, COORDINATE_OF_Y],
                                             NUM_OF_BINS, RANGE_OF_BINS)
    else:
        im_eq = lookup_table.flat[(im_orig * MAX_GRAY_VAL).astype(int)] \
                / MAX_GRAY_VAL
        hist_eq, edge_of_bins = np.histogram(im_eq, NUM_OF_BINS, RANGE_OF_BINS)

    im_eq = np.clip(im_eq, 0, 1)

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):

    """
    This function performs optimal quantization of a given grayscale or
    RGB image.
    :param im_orig: the input image.
    :param n_quant: the number of quants needed.
    :param n_iter:  the number of iterations needed.
    :return: the quantized image.
    """

    is_grayscale = len(im_orig.shape) == LEN_OF_GRAYSCALE_SHAPE
    quantization_function = np.zeros(NUM_OF_BINS, FLOAT64)
    num_of_zeds = n_quant + 1
    values_of_q = np.zeros(n_quant, np.int64)
    values_of_z = np.zeros(num_of_zeds, np.int64)
    calculated_errors = []

    if is_grayscale:
        target_pic = im_orig
        target_pic = unnormalize_pic(target_pic)
    else:
        temp_yiq = rgb2yiq(im_orig)
        target_pic = temp_yiq[:, :, COORDINATE_OF_Y]
        target_pic = unnormalize_pic(target_pic)

    # Find histogram:
    pic_histogram, edge_of_bins = np.histogram(target_pic, NUM_OF_BINS, [0,
                                                                         255])

    edge_of_bins = np.rint(edge_of_bins)

    # find cumulative histogram:
    pic_cumulative = np.cumsum(pic_histogram)

    # Calculate the average number of pixels per quant:
    initial_partition_num = int(pic_cumulative[MAX_GRAY_VAL] / n_quant)

    initialize_zeds(pic_cumulative, values_of_z, n_quant,
                    initial_partition_num)

    for j in range(n_iter):

        find_values_of_q(n_quant, values_of_z, pic_histogram, values_of_q)

        prev_zeds = values_of_z.copy()

        find_values_of_zeds(values_of_q, n_quant, values_of_z)

        values_of_z = np.rint(values_of_z)

        find_the_error_values(pic_histogram, values_of_z, n_quant,
                              values_of_q, calculated_errors)

        if check_convergence(prev_zeds, values_of_z):
            break

    # Calculate the quantization function:
    for q in range(n_quant):
        start = np.floor(values_of_z[q])
        end = np.ceil(values_of_z[q + 1])
        start = int(start)
        end = int(end)

        if end == VALUE_OF_LAST_Z:
            end += 1

        quantization_function[start: end] = values_of_q[q]

    target_pic = np.round(np.interp(target_pic, edge_of_bins[
                                                :LAST_ELEMENT_INDEX],
                                    quantization_function))

    im_quant = target_pic.astype(FLOAT64)

    im_quant = im_quant / FACTOR_OF_NORMALIZATION

    if not is_grayscale:

        temp_yiq[:, :, COORDINATE_OF_Y] = im_quant
        im_quant = yiq2rgb(temp_yiq)

    return [im_quant, calculated_errors]


def check_convergence(prev_zeds, values_of_z):

    """
    This function checks if there is a convergence in the quantization
    algorithm.
    :param prev_zeds: previous values of zeds.
    :param values_of_z: current values of zeds.
    :return: True if there is a convergence in the quantization algorithm,
    False otherwise.
    """

    return np.array_equal(prev_zeds, values_of_z)


def find_the_error_values(pic_histogram, values_of_z, num_of_quants,
                          values_of_q, calc_errors):

    """
    This function calculates the error value according to the formula
    shown in Tirgul 1.
    :param pic_histogram: the histogram of the image.
    :param values_of_z: the current values of zeds.
    :param num_of_quants: the number of quants needed.
    :param values_of_q: the current values of cues.
    :param calc_errors: the errors list.
    """

    temp_err = 0
    for j in range(num_of_quants):
        start = values_of_z[j]
        end = values_of_z[j + 1]

        start = int(start)
        end = int(end)

        diff = values_of_q[j] - np.arange(start, end)

        temp_err = np.sum(diff * diff * pic_histogram[start: end]) + temp_err
    calc_errors.append(temp_err)

    return


def find_values_of_zeds(values_of_q, num_of_quants, values_of_z):

    """
    This function calculates the next values of zeds, according to the
    quantization algorithm.
    :param values_of_q: the current values of cues.
    :param num_of_quants: the number of quants needed.
    :param values_of_z: the current values of zeds.
    """

    for i in range(0, num_of_quants - 1):
        values_of_z[i + 1] = (values_of_q[i + 1] / 2) + (values_of_q[i] / 2)
        values_of_z[i + 1] = np.ceil(values_of_z[i + 1])

    return


def initialize_zeds(pic_cumulative, values_of_z, n_quant,
                    initial_partition_num):

    """
    This function calculates the initial values of zeds, according to the
    quantization algorithm.
    :param pic_cumulative: the cumulative histogram of the image.
    :param values_of_z: the current values of zeds.
    :param n_quant: the number of quants needed.
    :param initial_partition_num: the average number of pixels per quant.
    """

    for j in range(1, n_quant):
        values_of_z[j] = np.argmax(pic_cumulative >= initial_partition_num * j)

    values_of_z[0] = VALUE_OF_FIRST_Z
    values_of_z[n_quant] = VALUE_OF_LAST_Z

    return


def unnormalize_pic(target_pic):

    """
    This function unnormalizes the input image.
    :param target_pic: the input image.
    :return: the unnormalized image.
    """

    return np.around(target_pic * FACTOR_OF_NORMALIZATION).astype(INT8)


def find_values_of_q(num_of_quants, values_of_z, pic_histogram, values_of_q):
    """
    This function calculates the current values of cues, according to the
    quantization algorithm.
    :param num_of_quants: the number of quants needed.
    :param values_of_z: the current values of zeds.
    :param pic_histogram: the histogram of the image.
    :param values_of_q: the current values of cues.
    """

    for j in range(num_of_quants):

        start = values_of_z[j]
        end = values_of_z[j + 1]

        start = int(start)
        end = int(end)

        values_of_q[j] = np.rint(np.sum(np.arange(start, end) *
                                        pic_histogram[start: end]) / np.sum(
            pic_histogram[start: end])).astype(np.int64)

    return


def quantize_rgb(im_orig, n_quant):
    """
    This function performs quantization for full color images.
    :param im_orig: the original image to be quantized.
    :param n_quant: the number of quants needed.
    :return: the quantized image.
    """

    depth = im_orig.shape[DEPTH_COORDS]
    col = im_orig.shape[COLS_COORDS]
    row = im_orig.shape[ROWS_COORDS]

    vectorize_pic = np.reshape(im_orig, (row * col, depth))
    kmeans = KMeans(n_clusters=n_quant)
    clustered_pic = kmeans.fit_predict(vectorize_pic)
    intensities = kmeans.cluster_centers_

    return np.reshape(intensities[clustered_pic], (row, col, depth))
