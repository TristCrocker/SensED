import cv2
import keyboard
from picamera2 import Picamera2, Preview
import time

picamLeft = Picamera2(0)
picamRight = Picamera2(1)
camera_configL = picamLeft.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
camera_configR = picamRight.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
picamLeft.configure(camera_configL)
picamRight.configure(camera_configR)

picamLeft.start_preview(Preview.QTGL)
picamRight.start_preview(Preview.QTGL)

picamLeft.start()
picamRight.start()

num = 0

while True:
	imgLeft = picamLeft.capture_array("main")
	imgRight = picamRight.capture_array("main")
	
	#if keyboard.is_pressed('q'):
	#	break
	#elif keyboard.is_pressed('s'):
	time.sleep(5)
	cv2.imwrite("/home/raspberry/SensED/images/stereoLeft/imageL" + str(num) + ".png", imgLeft)
	cv2.imwrite("/home/raspberry/SensED/images/stereoRight/imageR" + str(num) + ".png", imgRight)
	print("Images saved")
	num+=1
	
	

