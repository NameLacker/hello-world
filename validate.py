# import keras
from PIL import Image
import csv
from keras.models import load_model
import numpy as np
import pandas as pd



test_data_dir = 'F:/AI/image_scene_training/'

def read_image(image_name):
    img = Image.open(image_name)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.asarray(img, dtype='float32')
    img = img / 255
    if(img.shape == (256, 256)):
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

list = pd.read_csv(test_data_dir+'list.csv')
model = load_model('model/test.h5')
with open('res/result.csv', 'w', newline='') as f:
    header = ['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2']
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(list)):
        file_name = list.iloc[i][0]
        feature = read_image(test_data_dir+'data/'+file_name+'.jpg')
        print(feature.shape)
        feature = feature.reshape((1, 256, 256, 3))
        res = model.predict(feature)
        res = res[0]
        index = res.argsort()[-3:][::-1]

        result_data = np.append(file_name, index)
        writer.writerow(result_data)

