import sys

sys.path.append("../../")
from keras.datasets import mnist
import keras
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils import plot_model
from MyUtils import callbacklist, file_abspath, location

(train_img, train_lab), (test_img, test_lab) = mnist.load_data()
print(train_img.ndim)
print(train_img.shape)
print(train_img.dtype)
digit = train_img[1]
plt.imshow(digit,cmap='binary')
plt.show()


my_slice = train_img[10:100]
print(my_slice)