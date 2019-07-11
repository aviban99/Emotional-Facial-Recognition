# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import keras
import time 
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50, preprocess_input
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

# set path to the Training, Validation and Test image files
train_data = "/content/drive/My Drive/Datasetcropped/Train"
valid_data = "/content/drive/My Drive/Datasetcropped/Valid"
test_data = "/content/drive/My Drive/Datasetcropped/Test"

# define different classes present
CATEGORIES = ["Angry", "Contemptuous", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
num_classes= len(CATEGORIES)
print(len(CATEGORIES))

# create training data in the form of (image, class)
training_data = []

def create_training_data():
    for category in CATEGORIES:  #do different expressions

        path = train_data + "/" + category # create path to diferrent expressions
        class_num = CATEGORIES.index(category)  # get the classification  (0 to 7). 

        for img in tqdm(os.listdir(path)):  # iterate over each image per expression
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                #new_array = cv2.resize(img_array, (256, 256))  # resize to normalize data size
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()

# create validation data in the form of (image, class)
val_data = []

def create_val_data():
    for category in CATEGORIES:  #do different expressions

        path = valid_data + "/" + category # create path to diferrent expressions
        class_num = CATEGORIES.index(category)  # get the classification  (0 to 7). 

        for img in tqdm(os.listdir(path)):  # iterate over each image per expression
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                #new_array = cv2.resize(img_array, (256, 256))  # resize to normalize data size
                val_data.append([img_array, class_num])  # add this to our val_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_val_data()

# create test data in the form of (image, class)
testing_data = []

def create_test_data():
    for category in CATEGORIES:  #do different expressions

        path = test_data + "/" + category # create path to diferrent expressions
        class_num = CATEGORIES.index(category)  # get the classification  (0 to 7). 

        for img in tqdm(os.listdir(path)):  # iterate over each image per expression
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                #new_array = cv2.resize(img_array, (256, 256))  # resize to normalize data size
                testing_data.append([img_array, class_num])  # add this to our testing_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_test_data()

# check the length of train, validation and test data
print(len(training_data))
print(len(val_data))
print(len(testing_data))

# shuffle the training and validation data
import random
random.shuffle(training_data)
random.shuffle(val_data)

# separate X and y for the train, validation and test data
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

# for training data
for features,label in training_data:
    X_train.append(features)
    y_train.append(label)
X_train = np.array(X_train)
y_train = keras.utils.to_categorical(y_train, num_classes)  # Convert class vectors to binary class matrices.

X_train = X_train/255.0

# for validation data
for features,label in val_data:
    X_val.append(features)
    y_val.append(label)
X_val = np.array(X_val)
y_val = keras.utils.to_categorical(y_val, num_classes)  # Convert class vectors to binary class matrices.

X_val = X_val/255.0

# for test data
for features, label in testing_data:
  X_test.append(features)
  y_test.append(label)
X_test = np.array(X_test)
y_test = keras.utils.to_categorical(y_test, num_classes)  # Convert class vectors to binary class matrices.

X_test = X_test/255.0

print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(X_train.shape)
print(y_train.shape)

# cnn model with 3 convolutional layers alongwith AveragePooling2D and two fully connected dense layers
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(256,256,3), padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(num_classes)) # num_classes represent thesize of output i.e. 8 in this case
model.add(Activation('softmax'))

# check the model summary for the parameters created
model.summary()

# use EarlyStopping as callback for stopping the unnecessary epochs
es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fit the model on training data and validate it on validation data
model.fit(X_train, y_train,
              batch_size=32,
              epochs=10,
              validation_data=(X_val, y_val),
              shuffle=True,
              callbacks=[es])

# use the model to predict labels for the test data
score = model.predict_classes(X_test)

# evaluate model for validation accuracy and validation loss
loss, accuracy = model.evaluate(X_val, y_val)
print("VALIDATION ACCURACY: ", accuracy)



# change y_test back to categorical form
y_test2 = [ np.argmax(item) for item in y_test]

# evaluate the test accuracy using sklearn metrics
from sklearn import metrics
acc= metrics.accuracy_score(y_test2, score)
print("TEST ACCURACY: ", acc)

