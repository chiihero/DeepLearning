import os
import tqdm
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2gray, rgb2lab,lab2rgb

inputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\color\\"

imgs = os.listdir(inputcolor)
num = len(imgs)
# num = 1000
# data = np.empty((num, 256, 256, 1), dtype=float)
# label = np.empty((num, 256, 256, 2), dtype=float)
# for i in tqdm.trange(0, num, desc='Task', ncols=100):
#     Img = img_to_array(load_img(inputcolor + imgs[i]))
#     Img = np.array(Img, dtype=float)
#     Img = rgb2lab(1.0 / 255 * Img)
#
#     greyImg = Img[:, :, 0]
#     greyImg = greyImg.reshape(256, 256, 1)
#     colorImg = Img[:, :, 1:] / 128
#     colorImg = colorImg.reshape(256, 256, 2)
#
#     data[i] = greyImg
#     label[i] = colorImg
#
# # np.savez('save_Img(data,label)',x=data,y=label)
# np.savez_compressed('save_Imgz',data)

data = np.empty((num, 128, 128, 3), dtype=float)
for i in tqdm.trange(0, num, desc='Task', ncols=100):
    Img = img_to_array(load_img(inputcolor + imgs[i]))
    Img = np.array(Img, dtype=float)
    Img = rgb2lab(1.0 / 255 * Img)
    Img = Img.reshape(128, 128, 3)
    data[i] = Img
# np.savez('save_Img(data,label)',x=data,y=label)
np.savez_compressed('save_Imgz',data)