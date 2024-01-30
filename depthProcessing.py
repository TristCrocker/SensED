import numpy as np
import cv2 as cv

#Function to produce depth map of images
def produceDepthMap(leftImgPath, rightImgPath):
    # Import two images
    leftImage = cv.imread(leftImgPath, cv.IMREAD_GRAYSCALE)
    rightImage = cv.imread(rightImgPath, cv.IMREAD_GRAYSCALE)

    # Stereo matching with block match algorithm
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=21)
    depthMap = stereo.compute(leftImage, rightImage)

    return depthMap