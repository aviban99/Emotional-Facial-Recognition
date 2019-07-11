# Emotional-Facial-Recognition
Objective of this project is to classify image based on different emotions.
Classified images from a research based image dataset RaFD, can be downloaded from [here](http://www.socsci.ru.nl:8180/RaFD2/RaFD).

# About the Dataset
Used only frontal face images. 
Consisted of images of size (640, 1240) each of 67 subjects captured in 8 different emotional expressions:
* Angry
* Contemptuous
* Disgusted
* Fearful
* Happy
* Neutral
* Sad
* Surprised

# Image Preprocessing
This is the most important step.

1. First of all we will extract faces from the images. For this we will use Haar cascade filter. Each resulting image is obtained in size (256, 256).

 The code for facecropping: [crop.py](https://github.com/aviban99/Emotional-Facial-Recognition/blob/master/crop.py)
 
2. Separate the dataset into different emotional expression folders. Now, its time to manually separate our dataset into: Train, Valid and Test in suitable ratio. The final directory template looks like:
  - Dataset
    - Train 
      - Angry
      - Contemptuous
      - Disgusted
      - Fearful
      - Happy
      - Neutral
      - Sad
      - Surprised
    - Valid
      - Angry
      - ....
      - Surprised
    - Test
      - Angry
      - ....
      - Surprised

3. Since, the original dataset contained only 536 images, we will manually increase the total number of images using various OpenCV tools like rotation, flip, etc. However, we will only enhance images in Train and Valid folders.

The code for dataset enhancement: [data_aug.py](https://github.com/aviban99/Emotional-Facial-Recognition/blob/master/data_aug.py)

# Creation of Training, Validation and Test data
Our data is present in the form of images. Inorder to train the model on the data, we first need to create labels for each class i.e. we need to bring our data in the form (image, class). Here class represents a number from 0 to 7 for different expressions.

Here is a snippet for this: 
##
    def create_training_data():
      for category in CATEGORIES:  
        path = train_data + "/" + category 
        class_num = CATEGORIES.index(category) 
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                #new_array = cv2.resize(img_array, (256, 256))  
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
For more refer: [efe_cnn.py](https://github.com/aviban99/Emotional-Facial-Recognition/blob/master/efe_cnn%20(1).py)
# CNN Model 
Create a Sequential model using Keras and tensorflow as backend.

Model has three Conv2D layers along with AveragePooling2D layers.
## 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256,256,3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

Model has two fully connected dense layers and a flatten layer.

For more refer: [efe_cnn.py](https://github.com/aviban99/Emotional-Facial-Recognition/blob/master/efe_cnn%20(1).py)

# Training the Model
First compile the model and then fit it onto training data and validate it on the validation data with suitable number of epochs. To get decent accuracy and prevent unnecessary epochs, use EarlyStopping from keras.callbacks.

For more refer: [efe_cnn.py](https://github.com/aviban99/Emotional-Facial-Recognition/blob/master/efe_cnn%20(1).py)
##  
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=10,
              validation_data=(X_val, y_val),
              shuffle=True,
              callbacks=EarlyStopping)
              
# Evaluation and Prediction
After training check the validation accuracy and decide whether or not to fine-tune the model. Then predict the classes for the test data and evaluate the test accuracy.
Also, go through the confusion matrix for predicted test results. 

For more refer: [efe_cnn.py](https://github.com/aviban99/Emotional-Facial-Recognition/blob/master/efe_cnn%20(1).py)

# Further Work
* Transfer Learning
* Classification using Clustering
* Alternative approach using keras ImageDataGenerator
