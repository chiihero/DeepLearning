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
# 加载数据
from PIL import Image
import numpy as np

inputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\color\\"
inputbw = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\black&white\\"


def load_data():
    imgs = os.listdir(inputcolor)
    num = len(imgs)
    data = np.empty((num, 256, 256, 3), dtype="float32")
    label = np.empty((num,))
    for i in range(num):
        img = Image.open(inputcolor + imgs[i])
        arr = np.asarray(img, dtype="float32")
        # arr.resize((256, 256, 3))
        data[i, :, :, :] = arr
        label[i] = 0
    return data, label


def get_session(gpu_fraction=0.7):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=2, input_shape=(256, 256, 1)))
    model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=2))
    model.add(layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=2))
    model.add(layers.Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(layers.Conv2D(256,(3,3),activation='relu',padding='same'))


    model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(layers.Conv2D(16,(3,3),activation='relu',padding='same'))
    model.add(layers.Conv2D(2,(3,3),activation='tanh',padding='same'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model




def train():

    data, label = load_data()
    print(data.shape)
    print(data[1][1][1])
    train_data = data[:5000]
    train_labels = label[:5000]
    test_data = data[5000:]
    test_labels = label[5000:]
    model = build_model()
    model_file = 'simple_model.h5'
    history = model.fit((train_data,train_labels), validation_data=(test_data, test_labels),
                                  epochs=50,
                                  callbacks=callbacklist)
    model.save(model_file)

    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


KTF.set_session(get_session())
train()