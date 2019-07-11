# import necessary libraries
import cv2
import os
import tqdm
import matplotlib.pyplot as plt

# create a function to crop faces from the image usinf Haarcascade filter
def facecrop(image):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + facedata)

    img = cv2.imread(os.path.join(dest,image))

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        cv2.imwrite(os.path.join(dest,fname+"_2_"+ext), sub_face)
       
    return

# set path to the folder containing images
dest = "/content/drive/My Drive/Datasetcropped/Images"

# call the facecrop function
for image in os.listdir(dest):
        facecrop(image)

