import os
import sys

sys.path.append("../../")
import keras
from keras import models, layers
from keras.utils import to_categorical
from MyUtils import callbacklist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from skimage.color import rgb2gray, rgb2lab,lab2rgb
from skimage.io import imsave
# 加载数据
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

inputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\color\\"
inputbw = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\black&white\\"
outputfile = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\netcolor\\"


def load_data():
    image = img_to_array(load_img(inputcolor+'000011.jpg'))
    image = np.array(image, dtype=float)
    train_data = rgb2lab(1.0 / 255 * image)[:, :, 0]
    train_labels = rgb2lab(1.0 / 255 * image)[:, :, 1:]
    train_labels /= 128
    train_data = train_data.reshape(1, 256, 256, 1)
    train_labels = train_labels.reshape(1, 256, 256, 2)
    return train_data,train_labels


def build_model():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(None, None, 1)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def train():
    train_data, train_labels = load_data()
    print(train_data.shape)
    model = build_model()
    model_file = 'simple_model.h5'
    history = model.fit(x=train_data, y=train_labels,
                        epochs=500,
                        batch_size=1
                       )
    model.save(model_file)





    print(model.evaluate(train_data, train_labels, batch_size=1))
    output = model.predict(train_data)
    output *= 128
    # Output colorizations
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = train_data[0][:,:,0]
    cur[:,:,1:] = output[0]
    imsave(outputfile+'1.jpg' ,lab2rgb(cur))
train()
