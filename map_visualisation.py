import cv2
import numpy as np
from skimage.measure import block_reduce


def display_disparity(disp, window_name, colour=False):
    # Disparity map contains extreme values that cannot be displayed
    # By normalising the disparity map, it becomes possible to display it
    normalised_image = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)

    # If colour is set to True then the output image will be in colour (rather than greyscale)
    if colour:
        normalised_image = cv2.applyColorMap(normalised_image, cv2.COLORMAP_JET)

    # Displays the disparity map in the specified window
    cv2.imshow(window_name, normalised_image)
    # Waits 1ms for a key to be pressed, then stores the key that is pressed
    key = cv2.waitKey(1)
    # If the pressed key is q then the program quits
    if key == ord('q'):
        quit()


def downsample_map(map_array, target_res):
    current_shape = map_array.shape
    # Presumes that the given depth map has the same aspect ratio as the grid of motors
    # For efficiency purposes, camera frame cropping should be done before disparity map generation
    block_size = (int((current_shape[0] / target_res[1]) / 10), int((current_shape[1] / target_res[0]) / 10))
    # Makes the value of each block the max value of its sub-blocks
    downsampled_map = block_reduce(map_array, block_size, func=np.median)
    block_size = (10, 10)
    # Since disparity is inversely proportional to depth, we want to find the max disparity values in order to get
    # minimum depth values.
    downsampled_map = block_reduce(downsampled_map, block_size, func=np.max, cval=-1)
    return downsampled_map


def upscale_map(downsampled_map, original_map):
    # Gets the shapes of both arrays and then finds the necessary block size
    down_res = downsampled_map.shape
    original_res = original_map.shape
    block_size = (int(original_res[0] / down_res[0]), int(original_res[1] / down_res[1]))
    # Scales the downsampled map back to the original size by copying array values n times along
    # axis 0 and m times along axis 1 for a given (n, m) block size
    scaled_map = np.kron(downsampled_map, np.ones(block_size))
    return scaled_map


# def remove_bad_edge(disp_map):
#     clean_disp = disp_map.copy()
#
#     # Deletes columns that have -1 as the value in the first row
#     bad_columns = clean_disp[0] == -1
#     print(bad_columns)
#     clean_disp = clean_disp[:, np.logical_not(bad_columns)]
#
#     # Ensures that the resolution remains divisible by 10
#     print(clean_disp.shape)
#     clean_disp = np.delete(clean_disp, slice((clean_disp.shape[1] // 10) * 10, clean_disp.shape[1]), axis=1)
#     print(clean_disp.shape)
#     return clean_disp

