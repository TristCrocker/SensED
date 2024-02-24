# Real time implementation in here
import math
import cv2 as cv

from depth_map import depthProcessing, map_visualisation
from motor_output import MotorOutputDemo1

# Retrieve calibration data
cvFile = cv.FileStorage()
cvFile.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cvFile.getNode('StereoMapL_x').mat()
stereoMapL_y = cvFile.getNode('StereoMapL_y').mat()
stereoMapR_x = cvFile.getNode('StereoMapR_x').mat()
stereoMapR_y = cvFile.getNode('StereoMapR_y').mat()

# Open cameras
capL = cv.VideoCapture(2)
capR = cv.VideoCapture(0)

stereo = depthProcessing.produceStereo()

i2c, drv = MotorOutputDemo1.initMotorVars()

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

# Real time loop
while capL.isOpened() and capR.isOpened():

    successL, frameL = capL.read()
    successR, frameR = capR.read()

    frameL = cv.resize(frameL, (640, 480))
    frameR = cv.resize(frameR, (640, 480))

    # Waits 1ms for a key to be pressed, then stores the key that is pressed
    key = cv.waitKey(1)
    # If the pressed key is q then the program quits
    if key == ord('q'):
        drv.realtime_value = 0
        break

    # Undistort and rectify
    frameR = cv.remap(frameR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frameL = cv.remap(frameL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # Convert frames to grayscale
    grayFrameR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
    grayFrameL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)

    cv.imshow("Right", grayFrameR)
    cv.imshow("Left", grayFrameL)

    # stereo, minDisparity, numDisparities = depthProcessing.produceParameterSliders(stereo, "Disparity Map")

    # Create and display full resolution disparity map
    dispMap, matcher = depthProcessing.produceDisparityMap(stereo, grayFrameL, grayFrameR)
    dispMap = map_visualisation.filter_map(dispMap, grayFrameL, matcher, grayFrameR)
    map_visualisation.display_disparity(dispMap, "Disparity Map")

    dispMapDown = map_visualisation.downsample_map(dispMap, (8, 8))
    map_visualisation.display_disparity(map_visualisation.upscale_map(dispMapDown, (480, 640)),
                                        "Disparity Down-sampled Map")
    depthMap = depthProcessing.produceDepthMap(dispMapDown)

    MotorOutputDemo1.motorOutput(depthMap, i2c, drv, a=1, c=3)

capL.release()
capR.release()
drv.realtime_value = 0
drv.mode = adafruit_drv2605.MODE_INTTRIG
cv.destroyAllWindows()
