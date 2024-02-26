#Real time implementation in here
from depth_map import depthProcessing
from depth_map import map_visualisation
from picamera2 import Picamera2, Preview
#import MotorOutputDemo1
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import keyboard
import math
from datetime import datetime
from objectDetection import objectDetection


#Retrieve calibration data
cvFile = cv.FileStorage()
cvFile.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cvFile.getNode('StereoMapL_x').mat()
stereoMapL_y = cvFile.getNode('StereoMapL_y').mat()
stereoMapR_x = cvFile.getNode('StereoMapR_x').mat()
stereoMapR_y = cvFile.getNode('StereoMapR_y').mat()

#Open cameras
picamLeft = Picamera2(0)
picamRight = Picamera2(1)

#camera_configL = picamLeft.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
#camera_configR = picamRight.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
#picamLeft.configure(camera_configL)
#picamRight.configure(camera_configR)

picamLeft.start_preview(Preview.QTGL)
picamRight.start_preview()

picamLeft.start()
picamRight.start()


stereo = depthProcessing.produceStereo()

net, classes = objectDetection.setupModel()

#i2c, drv = MotorOutputDemo1.initMotorVars()

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

#Sigmoid depth map nromalizing function
sigmoid = lambda x:1/(1+math.e**(1*(x/1000)-3))
t0 = datetime.now()
#Real time loop
while True:
    
    img = picamLeft.capture_array('main')
    

    t1 = datetime.now()
    time_passed = (t1-t0)
    print("Average FPS: " + str(1/time_passed.total_seconds()))
    t0 = t1

    imgLeft = picamLeft.capture_array("main")
    
    
    
    imgRight = picamRight.capture_array("main")

    if keyboard.is_pressed('q'):
        #drv.realtime_value = 0
        break

        
    #Undistort and rectify
    frameR = cv.remap(imgRight, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frameL = cv.remap(imgLeft, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    #Convert frames to grayscale
    grayFrameR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
    grayFrameL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
    
    
    
    
    cv.imshow("Right", grayFrameR)
    cv.imshow("Left", grayFrameL)

    #stereo, minDisparity, numDisparities = depthProcessing.produceParameterSliders(stereo, "Disparity Map")
    
    # Create and display full resolution disparity map
    dispMap, matcher = depthProcessing.produceDisparityMap(stereo, grayFrameL, grayFrameR)
    dispMap = map_visualisation.filter_map(dispMap, grayFrameL, matcher, grayFrameR)
    map_visualisation.display_disparity(dispMap, "Disparity Map")

    dispMapDown = map_visualisation.downsample_map(dispMap, (8, 8))
    map_visualisation.display_disparity(map_visualisation.upscale_map(dispMapDown, (640, 480)), "Disparity Down-sampled Map")
    depthMap = depthProcessing.produceDepthMap(dispMapDown)
    
    try:
        x,y = objectDetection.detectObject(img, net , classes)
    except:
        x = -1
        y = -1
    print(x,y)
    
    
    #kernel = np.array([1.8,2.0,1.0])
    
    #map_visualisation.display_disparity(dispMapDown, "Disparity Map")
    #map_visualisation.display_disparity(dispMapGaus, "Disp Map Real")
    
    
    
    
    
    
    #depthMap = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, depthMap)
    #depthMap = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, depthMap)

    
    
    #MotorOutputDemo1.motorOutput(depthMap, i2c, drv, a=1, c=3)
    
    #index = np.unravel_index(depthMap.argmax(), depthMap.shape)
    #print(depthMap[index[0], index[1]])
    
    #plt.imshow(depthMap, cmap="hot", interpolation='nearest')
    #plt.show()
    


    #Downsample disparity map and display
    # downsampledMap = map_visualisation.downsample_map(dispMap)
    # map_visualisation.display_disparity(downsampledMap, "Downsampled Disparity Map")

    #Produce depth for motor input
    # depthMap = depthProcessing.produceDepthMap(downsampledMap, 1, 1)
capL.release()
capR.release()
drv.realtime_value = 0
#drv.mode = adafruit_drv2605.MODE_INTTRIG
cv.destroyAllWindows()
