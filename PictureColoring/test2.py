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

inputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\test\\"
inputbw = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\black&white\\"
outputfile = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\netcolor\\"


# Get images
X = []
for filename in os.listdir(inputcolor):
    X.append(img_to_array(load_img(inputcolor+filename)))
X = np.array(X, dtype=float)

Xtrain = 1.0/255*X
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(256, 256, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model
model.fit_generator(image_a_b_gen(batch_size), epochs=5, steps_per_epoch=1)

color_me = []
for filename in os.listdir(inputcolor):
    color_me.append(img_to_array(load_img(inputcolor+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave(outputfile+str(i)+".png", lab2rgb(cur))