import numpy as np
import cv2 as cv
import glob




################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (7,4)
frameSize = (640,480) #change??



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 31 #change
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = []

imagesLeft = sorted(glob.glob('/home/raspberry/SensED/images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('/home/raspberry/SensED/images/stereoRight/*.png'))
for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    imgL = cv.resize(imgL, (640, 480))
    imgR = cv.resize(imgR, (640, 480))
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    print(retR)
    # If found, add object points, image points (after refining them)
    if (retL and retR) == True:

        objpoints.append(objp)
        
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


############## sv calibration #####################################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC | cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2|cv.CALIB_FIX_K3|cv.CALIB_FIX_K4|cv.CALIB_FIX_K5|cv.CALIB_FIX_K6|cv.CALIB_RATIONAL_MODEL|cv.CALIB_USE_INTRINSIC_GUESS|cv.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K1|cv.CALIB_FIX_K2|cv.CALIB_FIX_K3|cv.CALIB_FIX_K4|cv.CALIB_FIX_K5|cv.CALIB_FIX_K6|cv.CALIB_RATIONAL_MODEL|cv.CALIB_USE_INTRINSIC_GUESS|cv.CALIB_FIX_PRINCIPAL_POINT
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL,distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags) #??

#  rectification #
rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL,newCameraMatrixR, distR, grayL.shape[::-1],rot,trans,rectifyScale,(0,0))
np.savetxt('projMatR.txt', projMatrixR)
np.savetxt('projMatL.txt', projMatrixL)
stereoMapL =cv.initUndistortRectifyMap(newCameraMatrixL,distL,rectL,projMatrixL,grayL.shape[::-1],cv.CV_16SC2)
stereoMapR =cv.initUndistortRectifyMap(newCameraMatrixR,distR,rectR,projMatrixR,grayR.shape[::-1],cv.CV_16SC2)
print("Saving Parameters")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('StereoMapL_x', stereoMapL[0])
cv_file.write('StereoMapL_y', stereoMapL[1])
cv_file.write('StereoMapR_x', stereoMapR[0])
cv_file.write('StereoMapR_y', stereoMapR[1])

cv_file.release()
