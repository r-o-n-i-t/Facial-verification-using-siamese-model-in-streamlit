import cv2
import numpy as np

#Loading HaarCascade classifier
face_cascade = cv2.CascadeClassifier("./HaarCascade/haarcascade_frontalface_default.xml")

#Define function to extract faces from image
def faceExtract(test_img):
    opencvImage = cv2.imread(test_img)
    faces=face_cascade.detectMultiScale(opencvImage,scaleFactor=1.3,minNeighbors=5)
    #Then we take the image if it contains only 1 face
    print(len(faces))
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            #extracting the region of interest
            roi = opencvImage[y:y+h, x:x+w]
            roi = cv2.resize(roi, (128, 128))
            arr = np.array(roi)
            return arr
print("Function made successfully!!!!")