import keras
from keras.utils import to_categorical
(train_img, train_lab), (test_img, test_lab) = keras.datasets.mnist.load_data()
test_img = test_img.reshape((10000, 28 * 28))
test_img = test_img.astype('float32') / 255
test_lab = to_categorical(test_lab)
model = keras.models.load_model('mnist.h5')
test_loss, test_acc = model.evaluate(test_img, test_lab)

print(test_loss)
print(test_acc)