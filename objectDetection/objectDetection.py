import cv2
import numpy as np
from picamera2 import Picamera2, Preview

def setupModel():
	net = cv2.dnn.readNet('/home/raspberry/SensED/objectDetection/yolov3-tiny.cfg', '/home/raspberry/SensED/objectDetection/yolov3-tiny.weights')

	classes = []
	with open("/home/raspberry/SensED/objectDetection/coco.names", "r") as f:
		classes = f.read().splitlines()

	return net, classes

'''
picamLeft = Picamera2(0)

picamLeft.start_preview()
picamLeft.start()
'''
def detectObject(queue, img, net, classes):
	while True:
		#img = picamLeft.capture_array("main")
		img = cv2.resize(img, (640,480))
		####Merging doesnt work with main
		#b, g, r, a = cv2.split(img)
		#img = cv2.merge([r, g, b])
		
		height, width, _ = img.shape
		
		blob = cv2.dnn.blobFromImage(img, 1/255, (640, 480), (0,0,0), swapRB=True, crop=False)
		net.setInput(blob)
		output_layers_names = net.getUnconnectedOutLayersNames()
		layerOutputs = net.forward(output_layers_names)

		boxes = []
		confidences = []
		class_ids = []
		center_x = -1
		center_y = -1

		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				
				if confidence > 0.2 and class_id == 0:
					center_x = int(detection[0]*width)
					center_y = int(detection[1]*height)
					
					#w = int(detection[2]*width)
					#h = int(detection[3]*height)

					#x = int(center_x - w/2)
					#y = int(center_y - h/2)

					#boxes.append([x, y, w, h])
					#confidences.append((float(confidence)))
					#class_ids.append(class_id)
				
		queue.put((center_x, center_y))
		return (center_x, center_y)
		
		
		'''indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

		if len(indexes)>0:
			for i in indexes.flatten():
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				confidence = str(round(confidences[i],2))
				color = colors[i]
				cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
				cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

		cv2.imshow('Image', img)
		key = cv2.waitKey(1)
		if key==27:
			break'''
'''
cap.release()
cv2.destroyAllWindows()

net, classes = setupModel()

while True:
	img = picamLeft.capture_array('main')
	try:
		x,y = detectObject(img, net , classes)
	except:
		x = -1
		y = -1	
	print(x,y)
'''


#def calculateDownSampleCoordinate():

