from skimage.color import rgb2gray, rgb2lab,lab2rgb
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from skimage.io import imsave
import os
import sys
import tqdm
import keras
inputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\color\\"
outputfile = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\netcolor\\"



def load_data():
    imgs = os.listdir(inputcolor)
    # num = len(imgs)
    num = 100
    data = np.empty((num, 256, 256,1), dtype=float)
    label = np.empty((num, 256, 256, 2), dtype=float)
    for i in tqdm.trange(0, num, desc='Task', ncols=100):
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

train_data, train_labels = load_data()

model =  keras.models.load_model('simple_model.model')
test_loss, test_acc=model.evaluate(train_data, train_labels)
print(test_loss)
print(test_acc)
output = model.predict(train_data)
output *= 128
# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = train_data[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    imsave(outputfile + str(i) + '.jpg', lab2rgb(cur))