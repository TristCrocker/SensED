import cv2
import keyboard
from picamera2 import Picamera2, Preview

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
		cv2.imwrite("/home/raspberry/SensED/images/stereoLeft/imageL" + str(num) + ".png", imgLeft)
		cv2.imwrite("/home/raspberry/SensED/images/stereoRight/imageR" + str(num) + ".png", imgRight)
		print("Images saved")
		num+=1
	
	

