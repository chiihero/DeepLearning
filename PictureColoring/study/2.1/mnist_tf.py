import sys
sys.path.append("../../")
import matplotlib.pyplot as plt
from Utils.callback_tf import callbacklist
from tensorflow import keras



(train_img,train_lab),(test_img,test_lab) = keras.datasets.mnist.load_data()
model = keras.models.Sequential()
model.add(keras.layers.Dense(512,activation="relu",input_shape=(28*28,)))
model.add(keras.layers.Dense(10,activation="softmax"))

opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=opt,
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

train_img = train_img.reshape((60000, 28 * 28))
train_img = train_img.astype('float32') / 255
test_img = test_img.reshape((10000, 28 * 28))
test_img = test_img.astype('float32') / 255

train_lab = keras.utils.to_categorical(train_lab)
test_lab = keras.utils.to_categorical(test_lab)

history = model.fit(train_img,train_lab,validation_data=(test_img,test_lab),epochs=200,batch_size=128,callbacks=callbacklist)

# model = keras.models.load_model('e:/model.hdf5')
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
