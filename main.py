#Real time implementation in here
import depthProcessing
import map_visualisation
import cv2 as cv

dispMapWindow = cv.namedWindow("Disparity Map") #Disparity Map window
downsampleMapWindow = cv.namedWindow("Downsampled Disparity Map")

#Real time loop
while True:
    #Create and display disparity map
    dispMap = depthProcessing.produceDisparityMap(img1, img2)
    map_visualisation.display_disparity(dispMap, dispMapWindow)

    #Downsample disparity map and display
    downsampledMap = map_visualisation.downsample_map(dispMap)
    map_visualisation.display_disparity(downsampledMap, downsampleMapWindow)

    #Produce depth for motor input
    depthMap = depthProcessing.produceDepthMap(downsampledMap, 1, 1)