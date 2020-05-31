############################## Imports ######################################

from skimage.color import rgb2gray
import numpy as np
from imageio import imread
from scipy.ndimage.filters import convolve
import sol5_utils
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Activation, Input, Add, Conv2D

########################## Global constants  ################################

FACTOR_OF_NORMALIZATION = 255
LEN_OF_GRAYSCALE_SHAPE = 2
RGB_REPRESENTATION = 2
GRAYSCALE_REPRESENTATION = 1
BIGGER_CROP_FACTOR = 3
SUBTRACTION_PIXEL_VALUE = 0.5
RATE_OF_TRAINING_SET = 0.8
ADAM_CONSTANT = 0.9
PI = np.pi
pic_cache = dict()
MEAN_SQUARED_ERR = "mean_squared_error"
FLOAT64_TYPE = np.float64


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


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    while True:

        random_pics = np.random.choice(filenames, batch_size)
        original = np.ones((batch_size, crop_size[0], crop_size[1], 1))
        corrupted = np.ones((batch_size, crop_size[0], crop_size[1], 1))

        for j in range(batch_size):
            temp_rand_pic_name = random_pics[j]
            if temp_rand_pic_name not in pic_cache:
                pic = read_image(temp_rand_pic_name, GRAYSCALE_REPRESENTATION)
                pic_cache[temp_rand_pic_name] = pic
            else:
                pic = pic_cache[temp_rand_pic_name]

            x_range_big = pic.shape[0] - BIGGER_CROP_FACTOR * crop_size[0]
            y_range_big = pic.shape[1] - BIGGER_CROP_FACTOR * crop_size[1]

            x_of_bigger_crop = np.random.randint(0, x_range_big)
            y_of_bigger_crop = np.random.randint(0, y_range_big)

            x_end_bigger = BIGGER_CROP_FACTOR * crop_size[0] + x_of_bigger_crop
            y_end_bigger = BIGGER_CROP_FACTOR * crop_size[1] + y_of_bigger_crop

            bigger_patch = pic[x_of_bigger_crop: x_end_bigger,
                               y_of_bigger_crop: y_end_bigger]

            bigger_patch_corrupted = corruption_func(bigger_patch)

            x_range = bigger_patch.shape[0] - crop_size[0]
            y_range = bigger_patch.shape[1] - crop_size[1]

            x_crop = np.random.randint(0, x_range)
            y_crop = np.random.randint(0, y_range)

            x_end = crop_size[0] + x_crop
            y_end = crop_size[1] + y_crop

            small_patch = bigger_patch[x_crop: x_end, y_crop: y_end]

            target = small_patch.reshape(crop_size[0], crop_size[1], 1)
            target = target - SUBTRACTION_PIXEL_VALUE

            source = bigger_patch_corrupted[x_crop: x_end, y_crop: y_end]. \
                reshape(crop_size[0], crop_size[1], 1)
            source = source - SUBTRACTION_PIXEL_VALUE

            corrupted[j] = source
            original[j] = target

        yield corrupted, original


def resblock(input_tensor, num_channels):
    first_convol = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    act_relu = Activation('relu')(first_convol)
    last_convol = Conv2D(num_channels, (3, 3), padding='same')(act_relu)
    input_add = Add()([input_tensor, last_convol])
    return Activation('relu')(input_add)


def build_nn_model(height, width, num_channels, num_res_blocks):
    tensor_input = Input(shape=(height, width, 1))
    first_convol = Conv2D(num_channels, (3, 3), padding='same')(tensor_input)
    act = Activation('relu')(first_convol)
    for _ in range(num_res_blocks):
        act = resblock(act, num_channels)
    second_convol = Conv2D(1, (3, 3), padding='same')(act)
    add = Add()([tensor_input, second_convol])
    return Model(inputs=tensor_input, outputs=add)


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    order_images_randomly = np.random.permutation(images)

    threshold = int(round(RATE_OF_TRAINING_SET * len(images)))

    crop_size = (model.input_shape[1], model.input_shape[2])

    train = order_images_randomly[:threshold]
    validation = order_images_randomly[threshold:]

    load_validation = load_dataset(validation, batch_size, corruption_func,
                                   crop_size)

    load_train = load_dataset(train, batch_size, corruption_func, crop_size)

    model.compile(optimizer=Adam(beta_2=ADAM_CONSTANT), loss=MEAN_SQUARED_ERR)
    model.fit_generator(load_train, validation_data=load_validation,
                        epochs=num_epochs, steps_per_epoch=steps_per_epoch,
                        validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    input_shape = (corrupted_image.shape[0], corrupted_image.shape[1], 1)
    tensor_in = Input(shape=input_shape)
    model_weights = base_model.get_weights()
    base = base_model(tensor_in)
    model = Model(inputs=tensor_in, outputs=base)
    model.set_weights(model_weights)
    reshape_pic = corrupted_image.reshape(input_shape)
    sub = reshape_pic - SUBTRACTION_PIXEL_VALUE
    return (np.clip(model.predict(sub[np.newaxis, ...])[0] +
                    SUBTRACTION_PIXEL_VALUE, 0, 1)[..., 0]).astype(
        FLOAT64_TYPE)


def add_gaussian_noise(image, min_sigma, max_sigma):

    return np.clip(((FACTOR_OF_NORMALIZATION * (np.random.normal(0,
                                                                 np.
                                                                 random.uniform
                                                                 (min_sigma,
                                                                  max_sigma),
                                                                 image.
                                                                 shape) + image
                                                )).round()) /
                   FACTOR_OF_NORMALIZATION, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):

    pics = sol5_utils.images_for_denoising()

    def func_corrupt(pic):

        return add_gaussian_noise(pic, 0, 0.2)

    if not quick_mode:
        m = build_nn_model(24, 24, 48, num_res_blocks)
        train_model(m, pics, func_corrupt, 100, 100, 5, 1000)
        return m
    else:
        m = build_nn_model(24, 24, 48, num_res_blocks)
        train_model(m, pics, func_corrupt, 10, 3, 2, 30)
        return m


def add_motion_blur(image, kernel_size, angle):
    return convolve(image, sol5_utils.motion_blur_kernel(kernel_size, angle))


def random_motion_blur(image, list_of_kernel_sizes):
    size_k = np.random.choice(list_of_kernel_sizes)
    pic_cor = (FACTOR_OF_NORMALIZATION * add_motion_blur(image, size_k,
                                                         np.random.uniform(
                                                                 0, PI))).\
        round()
    clipped = np.clip(pic_cor / FACTOR_OF_NORMALIZATION, 0, 1)
    return clipped


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):

    pics = sol5_utils.images_for_deblurring()

    def func_random_blur(pic):

        return random_motion_blur(pic, [7])

    if not quick_mode:
        m = build_nn_model(16, 16, 32, num_res_blocks)
        train_model(m, pics, func_random_blur, 100, 100, 10, 1000)
        return m
    else:
        m = build_nn_model(16, 16, 32, num_res_blocks)
        train_model(m, pics, func_random_blur, 10, 3, 2, 30)
        return m
