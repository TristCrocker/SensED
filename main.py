# Real time implementation in here

import math
import time
from datetime import datetime

import cv2 as cv
import keyboard
import numpy as np
from picamera2 import Picamera2, Preview

from motor_output import motor_output
from depth_map import depthProcessing
from depth_map import map_visualisation
from objectDetection import objectDetection

projMatR = np.loadtxt('camera_images/projMatR.txt')
projMatL = np.loadtxt('camera_images/projMatL.txt')

# Parameter sliders
win = cv.namedWindow('Disparity Map')

# Retrieve calibration data
cvFile = cv.FileStorage()
cvFile.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cvFile.getNode('StereoMapL_x').mat()
stereoMapL_y = cvFile.getNode('StereoMapL_y').mat()
stereoMapR_x = cvFile.getNode('StereoMapR_x').mat()
stereoMapR_y = cvFile.getNode('StereoMapR_y').mat()

# Open cameras
picamLeft = Picamera2(0)
picamRight = Picamera2(1)

camera_configL = picamLeft.create_video_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="main")
camera_configR = picamRight.create_video_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="main")

picamLeft.configure(camera_configL)
picamRight.configure(camera_configR)

picamLeft.start()
picamRight.start()

time.sleep(2)

stereo = depthProcessing.produceStereo()

net, classes = objectDetection.setupModel()

# Sigmoid depth map normalizing function
sigmoid = lambda x: 1 / (1 + math.e ** (1 * (x / 1000) - 3))
t0 = datetime.now()

# Real time loop
while True:
    imgLeft = picamLeft.capture_array()
    imgRight = picamRight.capture_array()

    try:
        x, y = objectDetection.detectObject(imgLeft, net, classes)
    except:
        x = -1
        y = -1

    t1 = datetime.now()
    time_passed = (t1 - t0)
    print("Average FPS: " + str(1 / time_passed.total_seconds()))
    t0 = t1

    if keyboard.is_pressed('q'):
        break

    # Undistort and rectify
    frameR = cv.remap(imgRight, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frameL = cv.remap(imgLeft, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # Convert frames to grayscale
    grayFrameR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
    grayFrameL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)

    # Create and display full resolution disparity map
    dispMap, matcher = depthProcessing.produceDisparityMap(stereo, grayFrameL, grayFrameR)
    dispMap = map_visualisation.filter_map(dispMap, grayFrameL, matcher, grayFrameR)
    map_visualisation.display_disparity(dispMap, "Disparity Map")

    dispMapDown, block_size = map_visualisation.downsample_map(dispMap, (4, 4))
    map_visualisation.display_disparity(map_visualisation.upscale_map(dispMapDown, (640, 480)),
                                        "Disparity Down-sampled Map")
    depthMap = depthProcessing.produceDepthMap(dispMapDown, projMatR, projMatL)

    # Calculate downsampled x and y for obj detection
    if x != -1:
        x = int(x / block_size[1])

    if y != -1:
        y = int(y / block_size[0])
    print(y, x)
    print('Position', y, x)
    finalArray = np.zeros((4, 4, 2))
    for row in range(len(depthMap[:, 0])):
        for col in range(len(depthMap[0, :])):
            intensity = depthMap[row, col]
            pattern = -1
            if (x != -1 and y != -1) and (y == row and x == col):
                pattern = 2
            finalArray[row, col, 0] = intensity
            finalArray[row, col, 1] = pattern

    controller = motor_output.PCA9685_Controller()
    controller.control_motors(finalArray)

drv.realtime_value = 0
picamLeft.stop()
picamRight.stop()
cv.destroyAllWindows()
