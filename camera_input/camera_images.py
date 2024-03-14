from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np
import os
from datetime import datetime
import keyboard

picamLeft = Picamera2(0)
picamRight = Picamera2(1)

picamLeft.start_preview(Preview.QTGL)
picamRight.start_preview(Preview.QTGL)

picamLeft.start()
picamRight.start()


num = 0

while True:
	
	imgLeft = picamLeft.capture_array("main")
	imgRight = picamRight.capture_array("main")
	
	if keyboard.is_pressed('q'):
		break
	elif keyboard.is_pressed('s'):
		cv2.imwrite("images/stereoLeft/imageL" + str(num) + ".png", imgLeft)
		cv2.imwrite("images/stereoRight/imageR" + str(num) + ".png", imgRight)
		print("Images saved")
		num+=1

picamLeft.stop()
picamRight.stop()
picamLeft.stop_preview()
picamRight.stop_preview()

