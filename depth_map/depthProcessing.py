import numpy as np
import cv2 as cv


def produceStereo():
    # Stereo matching with block match algorithm
    stereo = cv.StereoSGBM_create(mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    
    numDisparities = 96
    blockSize = 11
    minDisparity = 0
    '''
    preFilterType = 1
    preFilterSize = 1*2 + 5
    preFilterCap = 31
    textureThreshold = 5
    uniquenessRatio = 15
    speckleRange = 0
    speckleWindowSize = 0
    disp12MaxDiff = 0
    minDisparity = 0'''

    '''stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)'''
    stereo.setMinDisparity(minDisparity)
    stereo.setNumDisparities(numDisparities)
    stereo.setMinDisparity(0)
    stereo.setSpeckleRange(1)
    stereo.setSpeckleWindowSize(200)
    stereo.setBlockSize(blockSize)
    stereo.setP1(8 * 3 * blockSize ** 2)
    stereo.setP2(32 * 3 * blockSize ** 2)
    return stereo


# Produce disparity map of images
def produceDisparityMap(stereo, leftRectifiedImage, rightRectifiedImage):

    

    # Compute disparity map and convert to 32 bit floating point
    disparity = stereo.compute(leftRectifiedImage, rightRectifiedImage)

    #disparity = disparity.astype(np.float32) / 16.0
    # disparity = (disparity/16.0 - minDisparity)/numDisparities

    return disparity, stereo


# Produce depth map from disparity map (Using camera parameters from calibration)
# baseline is distance between camera
def produceDepthMap(disparityMap, projMatR, projMatL):
    # Retrieve projection matrices
    

    # Decompose projection matrices
    kLeft, rLeft, tLeft = decomposeProjectionMatrix(projMatL)
    kRight, rRight, tRight = decomposeProjectionMatrix(projMatR)

    # Calculate focal length
    focalLength = kLeft[0][0]

    # Calculate baseline
    baseline = tRight[0] - tLeft[0]

    # Calculate depth map
    disparityMap[disparityMap == 0] = 0.1  # Remove all instances of zero, avoiding division by zero
    disparityMap[disparityMap == -1.0] = 0.1  # Remove all instances of zero, avoiding division by zero
    depthMap = focalLength * baseline / disparityMap  # Calulcate depth map from disparity map (Z=B*f/disparity)

    return depthMap


def decomposeProjectionMatrix(matrix):
    k, r, t, _, _, _, _ = cv.decomposeProjectionMatrix(matrix)
    t = (t / t[3])[:3]

    return k, r, t


def produceParameterSliders(stereo, window):
    numDisparities = cv.getTrackbarPos('numDisparities', window)
    blockSize = cv.getTrackbarPos('blockSize', window)
    #preFilterType = cv.getTrackbarPos('preFilterType', window)
    #preFilterSize = cv.getTrackbarPos('preFilterSize', window) * 2 + 5
    preFilterCap = cv.getTrackbarPos('preFilterCap', window)
    #textureThreshold = cv.getTrackbarPos('teztureThreshold', window)
    uniquenessRatio = cv.getTrackbarPos('uniqueness', window)
    speckleRange = cv.getTrackbarPos('speckleRange', window)
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize', window)
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', window)
    minDisparity = cv.getTrackbarPos('minDisparity', window)

    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    #stereo.setPreFilterType(preFilterType)
    #stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    #stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    return stereo, minDisparity, numDisparities
