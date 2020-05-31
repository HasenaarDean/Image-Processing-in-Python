############################## Imports ######################################
import numpy as np
import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy.ndimage.interpolation import map_coordinates
import sol4_utils

########################## Global constants  ################################
DERIVATIVE_ARRAY = np.array([[1, 0, -1]])
DERIVATIVE_ARRAY_TRANS = DERIVATIVE_ARRAY.T
K = 0.04
SIZE_OF_KERNEL = 3
SIZE_OF_WINDOW = 7
SIZE_OF_RADIUS = 3
FIRST_LAYER_OF_PYRAMID = 0
LAST_LAYER_OF_PYRAMID = 2
AXES_DOT_PRODUCT_ARRAY = [[1, 2], [1, 2]]
ROWS = 0
COLS = 1
BEST_INDEX = -2
FLAT_TO_ONE_DIM = (1, -1)
FIRST_COL = 0
LAST_COL = 2
NUM_OF_HOM_POINTS_NEEDED = 2
SIZE_OF_HOMOGRAPHY = 3
IDENTITY_HOMOGRAPHY = np.eye(SIZE_OF_HOMOGRAPHY)
ZERO_MATRIX = np.array([[0.,  0.,  0.], [0.,  0.,  0.], [0.,  0.,  0.]])
BOUNDING_BOX_SHAPE = 2
TYPE_INT_64 = "int64"

############################# Functions #####################################


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y]
    coordinates of the ith corner points.
    """
    x_derivative_image = convolve2d(im, DERIVATIVE_ARRAY, mode='same',
                                    boundary='symm')
    y_derivative_image = convolve2d(im, DERIVATIVE_ARRAY_TRANS, mode='same',
                                    boundary='symm')
    x_2_derivative_image = sol4_utils.blur_spatial(x_derivative_image ** 2,
                                                   SIZE_OF_KERNEL)
    y_2_derivative_image = sol4_utils.blur_spatial(y_derivative_image ** 2,
                                                   SIZE_OF_KERNEL)
    multiply_x_y = sol4_utils.blur_spatial(x_derivative_image *
                                           y_derivative_image, SIZE_OF_KERNEL)
    multiply_x_y_2 = multiply_x_y * multiply_x_y
    multiply_squares = x_2_derivative_image * y_2_derivative_image
    sum_of_x_y_squares = x_2_derivative_image + y_2_derivative_image
    sum_squares_2 = sum_of_x_y_squares * sum_of_x_y_squares
    response_image_r = multiply_squares - multiply_x_y_2 - K * sum_squares_2
    local_maximum_points_of_r = non_maximum_suppression(response_image_r)
    non_zero_indices = np.argwhere(local_maximum_points_of_r)

    return np.fliplr(non_zero_indices)


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y]
    coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at
    desc[i,:,:].
    """

    n_constant = pos.shape[0]
    k_constant = 1 + 2 * desc_rad
    temp_window_index = 0

    windows_of_desc = np.ones((n_constant, k_constant, k_constant)).\
        astype(np.float64)

    for point in pos:

        start_col_index = point[0] - desc_rad
        end_col_index = 1 + desc_rad + point[0]

        start_row_index = point[1] - desc_rad
        end_row_index = 1 + desc_rad + point[1]

        temp_window_coords = np.array([(x, y) for x in np.arange(
            start_row_index, end_row_index) for y in np.arange(start_col_index,
                                                               end_col_index)])

        trans_window = temp_window_coords.transpose()

        sampled_descriptor_matrix = map_coordinates(im, trans_window, order=1,
                                                    prefilter=False)

        mean = np.mean(sampled_descriptor_matrix)
        mean_sub = sampled_descriptor_matrix - mean
        norm = np.linalg.norm(mean_sub)

        if norm == 0:
            final_descriptor_matrix = np.zeros(sampled_descriptor_matrix.shape)
        else:
            final_descriptor_matrix = mean_sub / norm

        windows_of_desc[temp_window_index, :, :] = final_descriptor_matrix.\
            reshape((k_constant, k_constant))
        temp_window_index += 1

    return windows_of_desc


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row
                found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """

    first_layer = pyr[FIRST_LAYER_OF_PYRAMID]
    first_layer_features = spread_out_corners(first_layer, SIZE_OF_WINDOW,
                                              SIZE_OF_WINDOW, SIZE_OF_RADIUS)
    last_layer_features = first_layer_features / 4
    last_layer = pyr[LAST_LAYER_OF_PYRAMID]
    ftr_desc_arr = sample_descriptor(last_layer, last_layer_features,
                                     SIZE_OF_RADIUS)

    return [first_layer_features, ftr_desc_arr]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices
                in desc1.
                2) An array with shape (M,) and dtype int of matching indices
                in desc2.
    """

    grades = np.tensordot(desc1, desc2, AXES_DOT_PRODUCT_ARRAY)
    grades_trans = grades.transpose()

    best_cols = np.sort(grades, axis=COLS)[:, BEST_INDEX]
    best_cols_1_dim = best_cols.reshape(FLAT_TO_ONE_DIM)

    best_rows = np.sort(grades, axis=ROWS)[BEST_INDEX, :]
    best_rows_1_dim = best_rows.reshape(FLAT_TO_ONE_DIM)

    best_grades_bool = min_score <= grades
    best_rows_bool = best_rows_1_dim <= grades
    best_cols_bool = (best_cols_1_dim <= grades_trans).transpose()

    return np.where(best_grades_bool & best_rows_bool & best_cols_bool)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates
    obtained from transforming pos1 using H12.
    """

    three_dim_pos_1 = np.stack([pos1[:, 0], pos1[:, 1], np.array(pos1.shape[0]
                                                                 * [1])])
    three_dim_pos_1_homog = np.dot(H12, three_dim_pos_1)
    normalize = three_dim_pos_1_homog[2, :]
    values_of_x = three_dim_pos_1_homog[0, :] / normalize
    values_of_y = three_dim_pos_1_homog[1, :] / normalize

    return np.stack([values_of_x, values_of_y]).T


def euclidean_distance(do_homog, points2):

    """
    This function calculates the euclidean distance.
    :param do_homog: homography.
    :param points2: points 2.
    :return: The euclidean distance.
    """

    diff = do_homog - points2
    calculate_norm = np.linalg.norm(diff, axis=1)
    return calculate_norm * calculate_norm


def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y]
    coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y]
    coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal
                    set of inlier matches found.
    """

    max_random_index_plus_one = points1.shape[0]
    best_choice = []
    best_choice_grade = 0

    for random_choice in range(num_iter):
        current_choice = np.random.randint(0, max_random_index_plus_one,
                                           size=NUM_OF_HOM_POINTS_NEEDED)

        while current_choice[0] == current_choice[1]:
            current_choice = np.random.randint(0, max_random_index_plus_one,
                                               size=NUM_OF_HOM_POINTS_NEEDED)

        current_p1_choice = points1[current_choice]
        current_p2_choice = points2[current_choice]
        homog = estimate_rigid_transform(current_p1_choice, current_p2_choice,
                                         translation_only)
        do_homog = apply_homography(points1, homog)

        distance = euclidean_distance(do_homog, points2)

        mark_matches = np.argwhere(inlier_tol > distance)
        current_grade = len(mark_matches)
        if best_choice_grade < current_grade:
            best_choice_grade = current_grade
            best_choice = mark_matches

    best_choice = best_choice.reshape(best_choice_grade, )
    best_p1_choice = points1[best_choice]
    best_p2_choice = points2[best_choice]
    normalized_homog = estimate_rigid_transform(best_p1_choice,
                                                best_p2_choice,
                                                translation_only)

    answer = [normalized_homog, best_choice]

    return answer


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates
    of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates
    of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    new_image = np.hstack((im1, im2))
    plt.figure()

    plt.imshow(new_image, cmap=sol4_utils.CONST_GRAY)

    n = len(points2)

    width = im1.shape[1]

    for i in range(n):

        x_1 = points1[i, 0]
        x_2 = width + points2[i, 0]
        y_1 = points1[i, 1]
        y_2 = points2[i, 1]

        if i not in inliers:

            plt.plot([x_1, x_2], [y_1, y_2], marker='o', ms=3, mfc='red',
                     c='blue', lw=.4)
        else:
            plt.plot([x_1, x_2], [y_1, y_2], marker='o', ms=3, mfc='red',
                     c='yellow', lw=.7)

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate
      system m
    """

    num_of_homogs = len(H_succesive)
    num_of_homogs = num_of_homogs + 1
    homographies_list = [ZERO_MATRIX] * num_of_homogs
    homographies_list[m] = IDENTITY_HOMOGRAPHY

    for homog_r in range(m + 1, num_of_homogs):

        inverse_hom = np.linalg.inv(H_succesive[homog_r - 1])
        homographies_list[homog_r] = np.dot(homographies_list[homog_r - 1],
                                            inverse_hom)
        normalize = homographies_list[homog_r][2, 2]
        homographies_list[homog_r] = homographies_list[homog_r] / normalize

    for homog_l in range(m - 1, -1, -1):

        homographies_list[homog_l] = np.dot(homographies_list[homog_l + 1],
                                            H_succesive[homog_l])

        normalize = homographies_list[homog_l][2, 2]
        homographies_list[homog_l] = homographies_list[homog_l] / normalize

    return homographies_list


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually
    warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """

    box = apply_homography(np.array([[0, 0], [0, h], [w, 0], [w, h]]),
                           homography)

    right_down = max(box[:, 0]), max(box[:, 1])
    left_up = min(box[:, 0]), min(box[:, 1])

    temp_res = np.array([left_up, right_down], dtype=TYPE_INT_64)
    res = temp_res.reshape(BOUNDING_BOX_SHAPE, BOUNDING_BOX_SHAPE)

    return res


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """

    box = compute_bounding_box(homography, image.shape[1], image.shape[0])

    arrange_x = np.arange(box[0, 0], box[1, 0])
    arrange_y = np.arange(box[0, 1], box[1, 1])

    x_values, y_values = np.array(np.meshgrid(arrange_x, arrange_y))

    warping_backwards = apply_homography(np.array([x_values.flatten(),
                                                   y_values.flatten()]).
                                         transpose(),
                                         np.linalg.inv(homography))

    new_pic = map_coordinates(image, [warping_backwards[:, 1],
                                      warping_backwards[:, 0]], order=1,
                              prefilter=False).reshape((box[1, 1] - box[0, 1],
                                                        box[1, 0] - box[0, 0]))

    return new_pic


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in
         range(3)])


def filter_homographies_with_translation(homographies,
                                         minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of
    translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the
    transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares
    method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first
    coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of
    corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of
    corresponding points from image 2.
    :param translation_only: whether to compute translation only.
    False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where
    True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector
    on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the
    image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y]
    coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (
                corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [
            os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i
            in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers among
            # matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 ,
            # points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs,
                                                           (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from
        each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate
        # system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[
                                                              i], self.w,
                                                          self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2,
                                    endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros(
            (number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the
        # input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None,
                              :]
            # homography warps the slice center to the coordinate system of
            # the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in
                              self.homographies]
            # we are actually only interested in the x coordinate of each
            # slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :,
                                      0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(
            np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:,
                             :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) *
                                      panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros(
            (number_of_panoramas, panorama_size[1], panorama_size[0], 3),
            dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:,
                              boundaries[0] - x_offset: boundaries[
                                                            1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom,
                boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the
        # left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few' \
                                       ' images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
