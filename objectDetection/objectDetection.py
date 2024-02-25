import cv2
import numpy as np
from picamera2 import Picamera2, Preview

def setupModel():
    net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()

    return net, classes

#picamLeft = Picamera2(0)

#picamLeft.start_preview(Preview.QTGL)

#picamLeft.start()

#font = cv2.FONT_HERSHEY_PLAIN
#colors = np.random.uniform(0, 255, size=(100, 3))

def detectObject(img, net, classes):
#while True:
    #img = picamLeft.capture_array("main")
    img = cv2.resize(img, (320,320))
    b, g, r, a = cv2.split(img)
    img = cv2.merge([r, g, b])
    
    height, width, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.2 and class_id == 0:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                print(center_x, center_y)
                #w = int(detection[2]*width)
                #h = int(detection[3]*height)

                #x = int(center_x - w/2)
                #y = int(center_y - h/2)

                #boxes.append([x, y, w, h])
                #confidences.append((float(confidence)))
                #class_ids.append(class_id)
    return center_x, center_y
    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

#    if len(indexes)>0:
#        for i in indexes.flatten():
#            x, y, w, h = boxes[i]
#            label = str(classes[class_ids[i]])
#            confidence = str(round(confidences[i],2))
#            color = colors[i]
#            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

#    cv2.imshow('Image', img)
#    key = cv2.waitKey(1)
#    if key==27:
#        break

#cap.release()
#cv2.destroyAllWindows()
