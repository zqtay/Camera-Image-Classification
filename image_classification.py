from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import numpy as np
import os
import random
import cv2
import time
from tqdm import tqdm

## Preparation

LOAD_PRETRAINED = False # Set to True if you wish to use a pre-trained model
data_path = r'C:\Users\User\Desktop\data' # Your project directory
categories = ['Apple','Orange'] # Labels of the training data, must be identical to their respective folder names
img_size = (28, 28) # Desired input image size for your neural network


## Training data creation

def prepare(image_path):
  img_array = cv2.imread(image_path ,cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, img_size)
  reshaped_array = new_array.reshape(*img_size, 1)
  return reshaped_array

def create_training_data():
    training_data = []
    for category in categories:
        class_label = np.zeros(len(categories))
        class_label[categories.index(category)] = 1
        for img in tqdm(os.listdir(f'{data_path}\{category}')):
            try:
                new_array = prepare(f'{data_path}\{category}\{img}')
                training_data.append([new_array, class_label])
            except Exception:
                pass
    random.shuffle(training_data)
    return training_data

if not LOAD_PRETRAINED:
    training_data = create_training_data()
    X = []
    y = []
    for features,label in training_data:
        X.append(features/255)
        y.append(label)
    X = np.array(X)
    y = np.array(y)


## Model creation

if not LOAD_PRETRAINED:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(BatchNormalization(axis=3, epsilon=0.00001))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(len(categories), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


## Model training

if not LOAD_PRETRAINED:
    time_now = time.time()
    tensorboard = TensorBoard(log_dir=f'{data_path}\logs\{time_now}')
    # Open cmd from the project folder and type: tensorboard --logdir = logs/ --host localhost --port 8088

    model.fit(X, y, batch_size=10, epochs=10, validation_split=0.2, callbacks=[tensorboard])

    plot_model(model, to_file=f'{data_path}\model_name_{time_now}.png')
    model.save(f'{data_path}\model_name_{time_now}.h5')


## Pre-trained model loading

if LOAD_PRETRAINED:
    model = load_model(f'{data_path}\pretrained_model_name.h5')


## Camera detection

def camera(device_num):
    camera = cv2.VideoCapture(device_num)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    CAMERA_STATUS = True

    while CAMERA_STATUS:
        _, frame = camera.read()
        # frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (170, 90), (470, 390), (240, 100, 0), 2)
        roi = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)[92:388,172:468] / 255 # Region of interest
        roi = cv2.resize(roi, img_size).reshape(-1, *img_size, 1)
        camera_prediction = model.predict([roi])[0]

        for i,category in enumerate(categories):
            cv2.putText(frame, f'{category}: {round(camera_prediction[i]*100,3)}',
                        (480, 110+(i*20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

        cv2.imshow('Camera',frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera.release()
            cv2.destroyAllWindows()
            CAMERA_STATUS = False

KEY = input("Press ENTER to start camera detection, Q to end the process.\n")
if KEY.lower() == '':
    camera(0)
elif KEY.upper() == 'Q':
    pass
