import cv2
import torch
import numpy as np
from torchvision import models, transforms
from picamera2 import Picamera2, Preview
import time

from PIL import Image

torch.backends.quantized.engine = 'qnnpack'

picamLeft = Picamera2(0)

picamLeft.start_preview(Preview.QTGL)


picamLeft.start()


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
		# read frame
        image = picamLeft.capture_array("main")
        

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

	# preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        
        top = list(enumerate(output[0].softmax(dim=0)))
        
        top.sort(key=lambda x: x[1], reverse=True) 
        for idx, val in top[:10]:
            print(f"{val.item()*100:.2f}%, Id: {idx}")

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0
