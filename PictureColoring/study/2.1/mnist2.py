import os
import sys
sys.path.append("../../")
from keras.datasets import mnist
import keras
import numpy as np
from keras import models, layers
from keras.utils import to_categorical
from MyUtils import callbacklist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

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

(train_img, train_lab), (test_img, test_lab) = mnist.load_data()
model = models.Sequential()


# design model
model.add(layers.Convolution2D(25, (5, 5), input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(50, (5, 5)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# train_img = np.expand_dims(train_img, axis=3)

train_img = train_img.reshape((60000, 28, 28, 1))
# train_img = train_img.astype('float32') / 255
test_img = test_img.reshape((10000, 28, 28, 1))
# test_img = test_img.astype('float32') / 255
train_lab = to_categorical(train_lab, num_classes=10)
test_lab = to_categorical(test_lab, num_classes=10)

datagen = ImageDataGenerator(
  rotation_range= 10,zoom_range= 0.1,
  width_shift_range = 0.1,height_shift_range = 0.1,
  horizontal_flip = False,
    vertical_flip = False)

datagen.fit(train_img)
data_iter = datagen.flow(train_img, train_lab, batch_size=128)
history = model.fit_generator(data_iter, validation_data=(test_img, test_lab), steps_per_epoch=len(train_img),epochs=50,
                              callbacks=callbacklist)



# history= model.fit(train_img, train_lab, validation_data=(test_img, test_lab), batch_size=128, epochs=50,callbacks=[tbCallBack, esCallBack, tnonCallBack,rpCallBack,mcCallBack])



# model = models.load_model('e:/model.hdf5')
# test_loss, test_acc = model.evaluate(test_img, test_lab)
#
# print(test_loss)
# print(test_acc)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
