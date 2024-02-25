import os
import requests
import tkinter
from imageai.Detection import ObjectDetection
import cv2
import random

modelTinyYOLOv3 = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5'

if not os.path.exists('yolo.h5'):
    r = requests.get(modelTinyYOLOv3, timeout=0.5)
    with open('yolo.h5', 'wb') as outfile:
        outfile.write(r.content)


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel()




peopleImages = os.listdir("people")
randomFile = peopleImages[random.randint(0, len(peopleImages) - 1)]

peopleOnly = detector.CustomObjects(person=True)
detectedImage, detections = detector.detectCustomObjectsFromImage(custom_objects=peopleOnly, output_type="array", input_image="people/{0}".format(randomFile), minimum_percentage_probability=30)
convertedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)
cv2.imshow(convertedImage)