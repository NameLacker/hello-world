# coding=UTF-8
import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D
import matplotlib.pyplot as plt

train_data_dir = 'F:/AI/text/image_scene_test_b_0515/'

def one_to_hot(label, c=20):
    return np.eye(c)[label.reshape(-1)].T

def read_image(image_name):
    img = Image.open(image_name)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.asarray(img, dtype='float32')
    img = img / 255
    if (img.shape == (256, 256)):
        X = np.empty((256, 256, 3), dtype='float32')
        X[:, :, 0] = img
        X[:, :, 1] = img
        X[:, :, 2] = img
        img = X
    nan = np.isnan(img)
    img[nan] = 0
    inf = np.isinf(img)
    img[inf] = 0
    return img

#  img = read_image('D:/Downloads/pic/image_scene_training/data/00d315dd-1c66-11e8-aaec-00163e025669.jpg')
#  print(np.max(img))

############# Callback Class ###############
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
############# Callback Class ################



history = LossHistory()


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(20))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


############### get data ##############

file_category = pd.read_csv(train_data_dir+'list.csv')
file_category = file_category.sample(frac=1)



stride = 500
for i in range(0, len(file_category), stride):
    if(i+stride >= len(file_category)):
        k = len(file_category)
    else:
        k = i + stride
    X_input = np.empty((k-i, 256, 256, 3), dtype='float32')
    y_input = np.empty((k-i, 20))
    for m, j in zip(range(5), range(i, k)):
        file_name = file_category.iloc[j][0]
        label = file_category.iloc[j][1]
        label = np.array(one_to_hot(label).reshape(20,))
        img = read_image(train_data_dir+'data/'+file_name+'.jpg')
        X_input[m, :, :, :] = img
        y_input[m, :] = label


    model.fit(X_input, y_input, batch_size=64, epochs=20, validation_split=0.3, callbacks=[history], shuffle=True)

file_category = []

history.loss_plot('epoch')
model.save('res/test.h5')
