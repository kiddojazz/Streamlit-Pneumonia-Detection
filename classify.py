import tensorflow as tf
import numpy as np
import io
import cv2
from PIL import Image
import base64
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model # used for loading model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array



#Load the model
print('Loading...')
model = load_model('Pneumonia_X-ray100.h5')
print("Model Loaded***")

def converter(value):
    if value == 0:
        return "NORMAL"
    else:
        return "PNEUMONIA"

def predict(image1): 
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    #image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels

    print('Input', image)
    print('Output_binary', yhat*100,'%');
    print('Output', converter(np.argmax(yhat)));

          
    return converter(np.argmax(yhat))

