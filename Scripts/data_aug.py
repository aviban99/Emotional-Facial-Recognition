# import necessary libraries
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rotate, rescale
import tensorflow as tf
from PIL import Image

# function for data augmentatioin
def data_aug(image):
  img = cv2.imread(os.path.join(path,image))       # read image from the path
  
  (h, w)= img.shape[:2]                            # image dimensions
  center = (w / 2, h / 2)                          # image center
  angle90 = 90                                     # 90 deg angle for rotation
  angle270 = 270                                   # 270 deg angle for rotation
  scale = 1.0                                      # maintaining original image dimensions
  
  # for flipping the image
  flip_1 = np.fliplr(img)
  flip_2 = np.flipud(img)
  
  # for rotating image
  M = cv2.getRotationMatrix2D(center, angle90, scale)
  rot90 = cv2.warpAffine(img, M, (h, w))
  N = cv2.getRotationMatrix2D(center, angle270, scale)
  rot270 = cv2.warpAffine(img, N, (h, w))
  
  # for grayscale image
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # for thresholded image
  ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
  ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)

  # for binary thresholded images
  img1 = cv2.medianBlur(img_gray,5).astype('uint8')
  ret,th1 = cv2.threshold(img1,150,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
  
  # for images with noise
  blur = cv2.blur(img,(5,5))
  
  # for saving the new images
  fname, ext = os.path.splitext(image)
  cv2.imwrite(os.path.join(path,fname+"_1_"+ext), flip_1)
  cv2.imwrite(os.path.join(path,fname+"_2_"+ext), flip_2)
  cv2.imwrite(os.path.join(path,fname+"_3_"+ext), rot90)
  cv2.imwrite(os.path.join(path,fname+"_4_"+ext), rot270)
  cv2.imwrite(os.path.join(path,fname+"_5_"+ext), thresh1)
  cv2.imwrite(os.path.join(path,fname+"_6_"+ext), thresh2)
  cv2.imwrite(os.path.join(path,fname+"_7_"+ext), thresh3)
  cv2.imwrite(os.path.join(path,fname+"_8_"+ext), img_gray)
  cv2.imwrite(os.path.join(path,fname+"_9_"+ext), blur)
  cv2.imwrite(os.path.join(path,fname+"_10_"+ext), th1)
  cv2.imwrite(os.path.join(path,fname+"_11_"+ext), th2)
  cv2.imwrite(os.path.join(path,fname+"_12_"+ext), th3)
  
  return

CATEGORIES = ["Angry", "Contemptuous", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
dest = "/content/drive/My Drive/Datasetcropped/"
for category in CATEGORIES:
  path = dest + "/" + category
  for image in os.listdir(path):
    data_aug(image)

