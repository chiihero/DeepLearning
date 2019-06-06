import os
import sys
import tqdm
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
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

inputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\color\\"
inputbw = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\black&white\\"
outputfile = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\netcolor\\"
batch_size=10
def get_session(gpu_fraction=0.7):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())



def load_data():
    imgs = os.listdir(inputcolor)
    # num = len(imgs)
    num = 100
    data = np.empty((num, 256, 256,1), dtype=float)
    label = np.empty((num, 256, 256, 2), dtype=float)
    for i in tqdm.trange(0, num, desc='Task', ncols=100):
        # Img = Image.open(inputcolor + imgs[i])
        Img = img_to_array(load_img(inputcolor + imgs[i]))
        Img = np.array(Img, dtype=float)
        Img = rgb2lab(1.0 / 255 * Img)

        greyImg = Img[:, :,0]
        greyImg = greyImg.reshape(256, 256, 1)

        colorImg = Img[:, :, 1:]/128
        colorImg = colorImg.reshape(256, 256, 2)

        data[i] = greyImg
        label[i] = colorImg

    return data, label


def build_model():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(None, None, 1)))
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
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.InputLayer(input_shape=(None, None, 1)))
    # model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    # model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def train():
    data, label = load_data()
    print(data.shape[0])
    slite= int(data.shape[0]*0.9)
    train_data = data[:slite]
    train_labels = label[:slite]
    test_data = data[slite:]
    test_labels = label[slite:]
    model = build_model()
    model_file = 'simple_model.model'

    def data_generator(data, targets, batch_size):
        batches = (len(data) + batch_size - 1) // batch_size
        while (True):
            for i in range(batches):
                X = data[i * batch_size: (i + 1) * batch_size]
                Y = targets[i * batch_size: (i + 1) * batch_size]
                yield (X, Y)

    history = model.fit_generator(generator=data_generator(train_data, train_labels, batch_size),
                                  validation_data=(test_data, test_labels),
                                  epochs=10,
                                  steps_per_epoch=10
                                  )
    model.save(model_file)
    # model = keras.models.load_model(model_file)
    # # 绘制训练 & 验证的准确率值
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # model.evaluate(train_data, train_labels)
    # output = model.predict(train_data)
    # output *= 128
    # # Output colorizations
    # for i in range(len(output)):
    #     cur = np.zeros((256, 256, 3))
    #     cur[:, :, 0] = train_data[i][:, :, 0]
    #     cur[:, :, 1:] = output[i]
    #     imsave(outputfile+str(i)+'.jpg', lab2rgb(cur))
train()
