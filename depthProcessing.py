import numpy as np
import cv2 as cv

#Produce disparity map of images
def produceDisparityMap(leftRectifiedImage, rightRectifiedImage):
    
    numDisparities = 16 #Disparities in BM
    blockSize = 21 #Block Size for BM

    # Stereo matching with block match algorithm
    stereo = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(leftRectifiedImage, rightRectifiedImage)

    return disparity

#Produce depth map from disparity map
def produceDepthMap(disparityMap, baseline, focalLength):

    depthMap = baseline*focalLength/disparityMap #Calulcate depth map (Z=B*f/disparity)
    return depthMap

