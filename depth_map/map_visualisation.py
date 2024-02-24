import math

import cv2
import numpy as np
from skimage.measure import block_reduce


def display_disparity(disp, window_name, colour=False):
    # If colour is set to True then the output image will be in colour (rather than greyscale)
    normalised_image = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if colour:
        normalised_image = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    # Displays the disparity map in the specified window
    cv2.imshow(window_name, normalised_image)
    # Waits 1ms for a key to be pressed, then stores the key that is pressed
    key = cv2.waitKey(1)
    # If the pressed key is q then the program quits
    if key == ord('q'):
        quit()


def downsample_map(map_array, target_res):
    current_shape = map_array.shape
    # optimally the initial resolution should be divisible by the target resolution, i.e. if the target res is 4x8,
    # the initial x length should be divisible by 4 and initial y length by 8.
    block_size = (math.ceil(current_shape[0] / target_res[1]), math.ceil(current_shape[1] / target_res[0]))
    # The value selected is the 90th percentile, which should pick up areas of high disparity (and therefore low depth),
    # but also not select outliers.
    downsampled_map = block_reduce(map_array, block_size, func=np.quantile, cval=-1, func_kwargs={'q': 0.9})
    return downsampled_map


def upscale_map(downsampled_map, target_rows_and_cols):
    # Gets the shapes of both arrays and then finds the necessary block size
    down_res = downsampled_map.shape
    block_size = (int(target_rows_and_cols[0] / down_res[0]), int(target_rows_and_cols[1] / down_res[1]))
    # Scales the downsampled map back to the original size by copying array values n times along
    # axis 0 and m times along axis 1 for a given (n, m) block size
    scaled_map = np.kron(downsampled_map, np.ones(block_size))
    return scaled_map


def filter_map(left_disp, left_image, left_matcher, right_image):
    # Parameters for filter
    sigma = 1.5
    lmbda = 8000.0

    # We need the right disparity map for WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    right_disp = right_matcher.compute(right_image, left_image)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    filtered_disp = wls_filter.filter(left_disp, left_image, None, disparity_map_right=right_disp,
                                      right_view=right_image)
    # Makes the filtered disparity map displayable
    filtered_disp = cv2.ximgproc.getDisparityVis(filtered_disp, None)
    return filtered_disp


def calculate_bar_length(disp):
    # Calculates the length of the black bar on the left, so that it can be cropped from the disparity map
    for i in range(0, disp.shape[1]):
        if disp[0, i] >= 0 or disp[-1, i] >= 0:
            return i
