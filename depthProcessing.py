import numpy as np
import cv2 as cv

#Produce disparity map of images
def produceDisparityMap(leftRectifiedImage, rightRectifiedImage):
    
    #Block matching parameters
    numDisparities = 16 #Disparities in BM
    blockSize = 21 #Block Size for BM

    # Stereo matching with block match algorithm
    stereo = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    #Compute disparity map and convert to 32 bit floating point
    disparity = stereo.compute(leftRectifiedImage, rightRectifiedImage).astype(np.float32)/16 

    return disparity

#Produce depth map from disparity map (Using camera parameters from calibration)
#baseline is distance between camera
def produceDepthMap(disparityMap, baseline, focalLength):

    #Catch division by zero
    try:
         depthMap = baseline*focalLength/disparityMap #Calulcate depth map from disparity map (Z=B*f/disparity)
    except:
        print("Division by zero error")
    
    return depthMap

