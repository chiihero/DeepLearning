from keras.callbacks import TensorBoard,EarlyStopping,TerminateOnNaN,ReduceLROnPlateau,ModelCheckpoint
import os
import sys
import tensorflow as tf

import keras.backend.tensorflow_backend as KTF


file_abspath = os.path.abspath(sys.argv[0])  # exe所在文件地址
location = os.path.dirname(file_abspath)  # exe所在文件夹目录地址

tbCallBack  = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
esCallBack=EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
tnonCallBack = TerminateOnNaN()
rpCallBack = ReduceLROnPlateau(monitor='val_acc', factor=0.2,patience=3, min_lr=0.0001)
mcCallBack = ModelCheckpoint(filepath=file_abspath[:-3]+'.model', monitor='val_acc', mode='auto', period=1,save_best_only=True)

callbacklist=[tbCallBack,esCallBack,tnonCallBack,rpCallBack,mcCallBack]

