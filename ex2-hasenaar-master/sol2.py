############################## Imports ######################################

from skimage.color import rgb2gray
import numpy as np
from imageio import imread
from scipy import signal
from scipy.ndimage.filters import convolve
from scipy.io import wavfile
from scipy.ndimage.interpolation import map_coordinates

########################## Global constants  ##################################

FACTOR_OF_NORMALIZATION = 255
LEN_OF_GRAYSCALE_SHAPE = 2
RGB_REPRESENTATION = 2
GRAYSCALE_REPRESENTATION = 1
SIZE_OF_COMPLEX128 = 2
ARRAY_OF_DERIVATION = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
ARRAY_OF_DERIVATION_TRANSPOSE = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
COMPLEX_TYPE = np.complex128
FLOAT_TYPE = np.float64
UNITY_ROOT_CONSTANT = np.pi * 2j
UNITY_ROOT_CONSTANT_NEG = -1 * np.pi * 2j
V = 1
U = 0
IDENTITY = 1
VANDER_RANGE = (-1, 1)

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


def stft(y, win_length=640, hop_length=160):

    """
    This function calculates the stft.
    """

    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):

    """
    This function calculates the istft.
    """
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):

    """
    This function calculates the phase vocoder.
    """

    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def DFT(signal):

    """
    This function calculates the discrete fourier transform.
    :param signal: the signal we get.
    :return: the discrete fourier transform of the signal.
    """

    vec2 = np.arange(0, signal.shape[0]) / signal.shape[0]

    vec1 = np.arange(0, signal.shape[0]).reshape((signal.shape[0], 1))

    exp = np.exp(vec1 * vec2 * UNITY_ROOT_CONSTANT_NEG)

    return np.dot(exp, signal)


def IDFT(fourier_signal):

    """
    This function calculates the discrete inverse fourier transform.
    :param fourier_signal: the signal we get.
    :return: the discrete inverse fourier transform of the signal.
    """

    vec2 = np.arange(0, fourier_signal.shape[0]) / fourier_signal.shape[0]

    vec1 = np.arange(0, fourier_signal.shape[0]).\
        reshape((fourier_signal.shape[0], 1))

    exp = np.exp(vec1 * vec2 * UNITY_ROOT_CONSTANT)

    return np.dot(exp, fourier_signal) / fourier_signal.shape[0]


def DFT2(image):

    """
    This function calculates the discrete fourier 2D transform.
    :param image: the image we get.
    :return: the 2D discrete fourier transform of the image.
    """

    first_dft = DFT(image).T

    # this is actually a composition of two DFTs.
    second_dft = DFT(first_dft)

    return second_dft.T


def IDFT2(fourier_image):

    """
    This function calculates the inverse discrete fourier 2D transform.
    :param fourier_image: the image we get.
    :return: the inverse 2D discrete fourier transform of the image.
    """

    first_idft = IDFT(fourier_image).T

    # this is actually a composition of two IDFTs.
    second_idft = IDFT(first_idft)

    return second_idft.T


def change_rate(filename, ratio):

    """
    This function changes the duration of an audio file by keeping the same
    samples, but changing the sample rate written in the file header.
    :param filename: the file's name.
    :param ratio: the new ratio needed.
    :return: the new audio (with changed ratio).
    """

    sample_rate, data = wavfile.read(filename)
    new_sample_rate = int(sample_rate * ratio)
    wavfile.write('change_rate.wav', new_sample_rate, data)
    return


def pad_with_zeros(n, num_of_rows_in_data, dft):

    """
    This function pads the dft sides with zeros.
    :param n: num of samples.
    :param num_of_rows_in_data: number of rows in data.
    :param dft: the dft we would like to pad.
    :return: the new dft (padded with zeros).
    """

    half_zero_pad_size = -0.5 * (num_of_rows_in_data - n)
    half_zero_pad_size = int(half_zero_pad_size)

    half_zero_pad = np.zeros([half_zero_pad_size, 1])
    return np.concatenate((half_zero_pad, dft, half_zero_pad))


def truncate_dft(n, num_of_rows_in_data, dft):

    """
    This function truncates the dft sides.
    :param n: num of samples.
    :param num_of_rows_in_data: number of rows in data.
    :param dft: the dft we would like to truncate.
    :return: the new dft (with sides truncated).
    """

    delete_index = 0.5 * (num_of_rows_in_data - n)
    delete_index = int(delete_index)

    delete_end = n + delete_index
    delete_end = int(delete_end)

    return dft[delete_index:delete_end]


def resize(data, ratio):

    """
    This function changes the number of samples by the given ratio.
    It calculates the 1D ndarray of the dtype of data representing
    the new sample points.
    :param data: the data we want to change its size.
    :param ratio: the new ratio needed.
    :return: the changed data with new number of samples by the given ratio.
    """

    num_of_rows_in_data = data.shape[0]
    dft = np.fft.fftshift(DFT(data.reshape(num_of_rows_in_data, 1)))

    n = num_of_rows_in_data
    n = n / ratio
    n = int(n)
    data_type = data.dtype

    if IDENTITY <= ratio:

        dft = truncate_dft(n, num_of_rows_in_data, dft)

        rows_num_of_dft = dft.shape[0]

        return IDFT(np.fft.ifftshift(dft)).real.reshape([rows_num_of_dft], ).\
            astype(data_type)
    else:

        dft = pad_with_zeros(n, num_of_rows_in_data, dft)

        rows_num_of_dft = dft.shape[0]
        return IDFT(np.fft.ifftshift(dft)).real.reshape([rows_num_of_dft], ).\
            astype(data_type)


def change_samples(filename, ratio):

    """
    This function changes the duration of an audio file by reducing the number
    of samples.
    :param filename: the file's name.
    :param ratio: the new ratio needed.
    :return: new audio with reduced number of samples.
    """

    sample_rate, data = wavfile.read(filename)
    new_data = resize(data, ratio)
    wavfile.write("change_samples.wav", sample_rate, new_data.astype(np.int16))
    return


def resize_spectrogram(data, ratio):

    """
    This function speeds up a WAV file, without changing the pitch,
    using spectrogram scaling. This is done by computing the spectrogram,
    changing the number of spectrogram columns, and creating back the audio.
    :param data: the data we have to change.
    :param ratio: the new ratio needed.
    :return: a new WAV file as explained above.
    """

    mat = stft(data)
    new_mat = []
    for temp_row in range(mat.shape[0]):
        new_mat.append(resize(mat[temp_row], ratio))

    new_mat = np.asarray(new_mat)

    return np.ceil(istft(new_mat)).astype(np.int16)


def resize_vocoder(data, ratio):

    """
    This function speedups a WAV file by phase vocoding its spectrogram.
    :param data: the data we have to change.
    :param ratio: the new ratio needed.
    :return: a new WAV file as explained above.
    """

    return np.rint(istft(phase_vocoder(stft(data), ratio))).astype(np.int16)


def calculate_shift_u_or_v(im, u_or_v):

    """
    This function calculates the fftshift of an image.
    :param im: the input image.
    :param u_or_v: 0 or 1.
    :return: the fftshift of an image.
    """

    start_index = int(np.floor(-0.5 * im.shape[u_or_v]))
    end_index = int(np.ceil(0.5 * im.shape[u_or_v]))

    return np.fft.fftshift(np.arange(start_index, end_index))


def calculate_square_sum(im, shift_u, shift_v, fourier_im, fourier_im_trans):

    """
    This function calculates the absolute square sum,
    according to the fourier derivative formula.
    :param im: the image we want to calculate its derivative.
    :param shift_u: the fft shift for u.
    :param shift_v: the fft shift for v.
    :param fourier_im: the image after DFT2.
    :param fourier_im_trans: the transpose image after DFT2.
    :return: the absolute square sum, according to the fourier derivative
    formula.
    """

    y_const = UNITY_ROOT_CONSTANT_NEG / im.shape[V]
    x_const = UNITY_ROOT_CONSTANT_NEG / im.shape[U]

    delta_y = IDFT2(np.multiply(shift_u, fourier_im_trans).transpose())
    delta_x = IDFT2(np.multiply(shift_v, fourier_im))

    delta_y = y_const * delta_y
    delta_x = x_const * delta_x

    abs_y = np.abs(delta_y)
    abs_x = np.abs(delta_x)

    return (abs_y ** 2) + (abs_x ** 2)


def fourier_der(im):

    """
    This function computes the magnitude of image derivatives using Fourier
    transform.
    :param im: the input image.
    :return: the magnitude of image derivatives using Fourier transform.
    """

    shift_v = calculate_shift_u_or_v(im, V)
    shift_u = calculate_shift_u_or_v(im, U)

    fourier_im = DFT2(im)
    fourier_im_trans = fourier_im.transpose()

    square_sum = calculate_square_sum(im, shift_u, shift_v, fourier_im,
                                      fourier_im_trans)

    return np.sqrt(square_sum).astype(FLOAT_TYPE)


def conv_der(im):

    """
    This function computes the magnitude of image derivatives.
    We derive the image in each direction separately (vertical and horizontal)
    using simple convolution with [0.5, 0, -0.5] as a row and column vectors,
    to get the two image derivatives. Next, we use these derivative images to
    compute the magnitude image.
    :param im: the input image.
    :return: he magnitude of image derivatives.
    """

    delta_y = convolve(im, ARRAY_OF_DERIVATION_TRANSPOSE)
    delta_x = convolve(im, ARRAY_OF_DERIVATION)

    abs_delta_y = np.abs(delta_y)
    abs_delta_x = np.abs(delta_x)

    temp_sum = (abs_delta_y ** 2) + (abs_delta_x ** 2)

    return np.sqrt(temp_sum)
