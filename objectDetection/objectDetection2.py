from deepface import DeepFace
from picamera2 import Picamera2, Preview
import cv2

# Start capturing video
picamLeft = Picamera2(0)

camera_configL = picamLeft.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="main")
picamLeft.configure(camera_configL)

picamLeft.start_preview()

picamLeft.start()


while True:
    # Capture frame-by-frame
    img = picamLeft.capture_array("main")
    img = cv2.resize(img, (640,480))
    
    res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    print(res[0]['dominant_emotion'])	
    # Convert frame to grayscale
#    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
#    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#    for (x, y, w, h) in faces:
#        # Extract the face ROI (Region of Interest)
#        face_roi = gray_frame[y:y + h, x:x + w]
#
#        # Resize the face ROI to match the input shape of the model
#        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
#
#        # Normalize the resized face image
#        normalized_face = resized_face / 255.0
#
#        # Reshape the image to match the input shape of the model
#        reshaped_face = normalized_face.reshape(1, 48, 48, 1)
#
#        # Predict emotions using the pre-trained model
#        preds = model.predict(reshaped_face)[0]
#        emotion_idx = preds.argmax()
#        emotion = emotion_labels[emotion_idx]
#
 #       # Draw rectangle around face and label with predicted emotion
 #       cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
 #       cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cv2.destroyAllWindows()
