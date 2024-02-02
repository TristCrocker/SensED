#Real time implementation in here
import depthProcessing
import map_visualisation
import cv2 as cv

cv.namedWindow("Disparity Map") #Disparity Map window
cv.namedWindow("Downsampled Disparity Map")

#Retrieve calibration data
cvFile = cv.FileStorage()
cvFile.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cvFile.getNode('stereoMapL_x').mat()
stereoMapL_y = cvFile.getNode('stereoMapL_y').mat()
stereoMapR_x = cvFile.getNode('stereoMapR_x').mat()
stereoMapR_y = cvFile.getNode('stereoMapR_y').mat()

#Open cameras
capL = cv.VideoCapture(0, cv.CAP_DSHOW)
capR = cv.VideoCapture(2, cv.CAP_DSHOW)

#Real time loop
while (capL.isOpened() and capR.isOpened()):

    successL, frameL = capL.read()
    successR, frameR = capR.read()

    #Undistort and rectify
    frameR = cv.remap(frameR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frameL = cv.remap(frameL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    #Create and display full resolution disparity map
    dispMap = depthProcessing.produceDisparityMap(frameL, frameR)
    map_visualisation.display_disparity(dispMap, "Disparity Map")


    #Downsample disparity map and display
    # downsampledMap = map_visualisation.downsample_map(dispMap)
    # map_visualisation.display_disparity(downsampledMap, "Downsampled Disparity Map")

    #Produce depth for motor input
    # depthMap = depthProcessing.produceDepthMap(downsampledMap, 1, 1)