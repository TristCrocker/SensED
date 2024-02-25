#Real time implementation in here
import math
from datetime import datetime
import multiprocessing

import cv2 as cv
import keyboard
from picamera2 import Picamera2, Preview

from depth_map import depthProcessing, map_visualisation
from motor_output import motor_output
from object_detection import objectDetection

# Retrieve calibration data
cvFile = cv.FileStorage()
cvFile.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cvFile.getNode('StereoMapL_x').mat()
stereoMapL_y = cvFile.getNode('StereoMapL_y').mat()
stereoMapR_x = cvFile.getNode('StereoMapR_x').mat()
stereoMapR_y = cvFile.getNode('StereoMapR_y').mat()

#Open cameras
picamLeft = Picamera2(0)
picamRight = Picamera2(1)

picamLeft.start_preview(Preview.QTGL)
picamRight.start_preview(Preview.QTGL)

picamLeft.start()
picamRight.start()

stereo = depthProcessing.produceStereo()

net, classes = objectDetection.setupModel()

'''
def nothing(x):
    pass

cv.createTrackbar('numDisparities', "Disparity Map",1,17,nothing)
cv.createTrackbar('blockSize', "Disparity Map",5,50,nothing)
cv.createTrackbar('preFilterType', "Disparity Map",1,1,nothing)
cv.createTrackbar('preFilterSize', "Disparity Map",2,25,nothing)
cv.createTrackbar('preFilterCap', "Disparity Map",5,62,nothing)
cv.createTrackbar('teztureThreshold', "Disparity Map",10,100,nothing)
cv.createTrackbar('uniqueness', "Disparity Map",15,100,nothing)
cv.createTrackbar('speckleRange', "Disparity Map",0,100,nothing)
cv.createTrackbar('speckleWindowSize', "Disparity Map",3,25,nothing)
cv.createTrackbar('disp12MaxDiff', "Disparity Map",5,25,nothing)
cv.createTrackbar('minDisparity', "Disparity Map",5,25,nothing)
'''

# Sigmoid depth map normalizing function
sigmoid = lambda x: 1 / (1 + math.e ** (1 * (x / 1000) - 3))
t0 = datetime.now()
# Real time loop
while True:

    t1 = datetime.now()
    time_passed = (t1 - t0)
    print("Average FPS: " + str(1 / time_passed.total_seconds()))
    t0 = t1

    imgLeft = picamLeft.capture_array("main")
    imgRight = picamRight.capture_array("main")

    if keyboard.is_pressed('q'):
        break

    # Undistort and rectify
    frameR = cv.remap(imgRight, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frameL = cv.remap(imgLeft, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # Convert frames to grayscale
    grayFrameR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
    grayFrameL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)

    cv.imshow("Right", grayFrameR)
    cv.imshow("Left", grayFrameL)

    # stereo, minDisparity, numDisparities = depthProcessing.produceParameterSliders(stereo, "Disparity Map")

    depth_process = multiprocessing.Process(target=depthProcessing.depth_processing, args=[stereo, grayFrameL, grayFrameR])
    object_process = multiprocessing.Process(target=objectDetection.detectObject, args=[imgLeft, net, classes])

    depth_process.start()
    object_process.start()

    depth_process.join()
    object_process.join()

    controller = motor_output.PCA9685_Controller()
    # TODO: create array with intensity and pattern for motors
    controller.control_motors()

drv.realtime_value = 0
cv.destroyAllWindows()
