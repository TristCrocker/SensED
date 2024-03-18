#Real time implementation in here
import time

from depth_map import depthProcessing
from depth_map import map_visualisation
import motor_output
import numpy as np
import cv2 as cv
import keyboard
import math
from imutils.video import VideoStream
from datetime import datetime
from objectDetection import objectDetection

projMatR = np.loadtxt('projMatR.txt')
projMatL = np.loadtxt('projMatL.txt')

#Parameter sliders
win = cv.namedWindow('Disparity Map')

#Retrieve calibration data
cvFile = cv.FileStorage()
cvFile.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cvFile.getNode('StereoMapL_x').mat()
stereoMapL_y = cvFile.getNode('StereoMapL_y').mat()
stereoMapR_x = cvFile.getNode('StereoMapR_x').mat()
stereoMapR_y = cvFile.getNode('StereoMapR_y').mat()

#Open cameras
picamLeft = VideoStream(src=0, usePiCamera=True, resolution=(640, 480)).start()
picamRight = VideoStream(src=1, usePiCamera=True, resolution=(640, 480)).start()

stereo = depthProcessing.produceStereo()

net, classes = objectDetection.setupModel()


#i2c, drv = MotorOutputDemo1.initMotorVars()

def nothing(x):
    pass
    
'''
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
cv.createTrackbar('minDisparity', "Disparity Map",5,25,nothing)'''


#Sigmoid depth map nromalizing function
sigmoid = lambda x:1/(1+math.e**(1*(x/1000)-3))
t0 = datetime.now()
#Real time loop

while True:

    imgLeft = picamLeft.read()
    imgRight = picamRight.read()
    
    try:
        x,y = objectDetection.detectObject(imgLeft, net , classes)
    except:
        x = -1
        y = -1
    #print(x,y)
    
    
    t1 = datetime.now()
    time_passed = (t1-t0)
    print("Average FPS: " + str(1/time_passed.total_seconds()))
    t0 = t1

    if keyboard.is_pressed('q'):
        #drv.realtime_value = 0
        break

    #Undistort and rectify
    frameR = cv.remap(imgRight, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frameL = cv.remap(imgLeft, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    #Convert frames to grayscale
    grayFrameR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
    grayFrameL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
    
    #cv.imshow("Right", grayFrameR)
    #cv.imshow("Left", grayFrameL)

    #stereo, minDisparity, numDisparities = depthProcessing.produceParameterSliders(stereo, "Disparity Map")
    
    # Create and display full resolution disparity map
    dispMap, matcher = depthProcessing.produceDisparityMap(stereo, grayFrameL, grayFrameR)
    #dispMap = cv.GaussianBlur(dispMap,(5,5),cv.BORDER_DEFAULT)
    dispMap = map_visualisation.filter_map(dispMap, grayFrameL, matcher, grayFrameR)
    map_visualisation.display_disparity(dispMap, "Disparity Map")
    
    #stereo, minDis, maxDisp = depthProcessing.produceParameterSliders(stereo, "Disparity Map")
    
    dispMapDown, block_size = map_visualisation.downsample_map(dispMap, (4, 4))
    map_visualisation.display_disparity(map_visualisation.upscale_map(dispMapDown, (640, 480)), "Disparity Down-sampled Map")
    depthMap = depthProcessing.produceDepthMap(dispMapDown, projMatR, projMatL)
    
    #Calculate downsampled x and y for obj detection
    if x != -1:
        x = int(x/block_size[1])

    if y != -1:
        y = int(y/block_size[0])
    print(y,x)    
    print('Position', y,x)
    finalArray = np.zeros((4, 4, 2))
    for row in range(len(depthMap[: ,0])):
        for col in range(len(depthMap[0 ,:])):
            intensity = depthMap[row, col]
            pattern = -1
            if (x != -1 and y != -1) and (y == row and x == col):
                pattern = 2
            finalArray[row, col, 0] = intensity
            finalArray[row, col, 1] = pattern
    #print(finalArray)

    controller = motor_output.PCA9685_Controller()
    controller.control_motors(finalArray)


    #kernel = np.array([1.8,2.0,1.0])
    
    #map_visualisation.display_disparity(dispMapDown, "Disparity Map")
    #map_visualisation.display_disparity(dispMapGaus, "Disp Map Real")
    
    
    
    
    
    #depthMap = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, depthMap)
    #depthMap = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, depthMap)

    
    
    #MotorOutputDemo1.motorOutput(depthMap, i2c, drv, a=1, c=3)
    
    #index = np.unravel_index(depthMap.argmax(), depthMap.shape)
    #print(depthMap[index[0], index[1]])
    
    #depthMapLarge = depthProcessing.produceDepthMap(dispMap, projMatR, projMatL)
    
    #plt.imshow(depthMapLarge, cmap="hot", interpolation='nearest')
    #plt.show()
    


    #Downsample disparity map and display
    # downsampledMap = map_visualisation.downsample_map(dispMap)
    # map_visualisation.display_disparity(downsampledMap, "Downsampled Disparity Map")

    #Produce depth for motor input
    # depthMap = depthProcessing.produceDepthMap(downsampledMap, 1, 1)

drv.realtime_value = 0
#drv.mode = adafruit_drv2605.MODE_INTTRIG
picamLeft.stop()
picamRight.stop()
cv.destroyAllWindows()
