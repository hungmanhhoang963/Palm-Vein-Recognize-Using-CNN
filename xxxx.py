import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
TRAIN_DIR = 'C:\\Users\\hungm\\OneDrive\\Desktop\\train_data\\train\\'
TEST_DIR = 'C:\\Users\\hungm\\OneDrive\\Desktop\\test_data\\test\\'
train_dogs = ['./train/{}'.format(i) for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = ['./train/{}'.format(i) for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images_dogs_cats = ['./test/{}'.format(i) for i in os.listdir(TEST_DIR)]
#len(train_dogs), len(train_cats), len(test_images_dogs_cats)
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

train_dogs.sort(key=natural_keys)
train_cats.sort(key=natural_keys)
train_images_dogs_cats = train_dogs[0:1300] + train_cats[0:1300] 
test_images_dogs_cats.sort(key=natural_keys)
print(train_images_dogs_cats[1333])