from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage import morphology, filters
import os.path
from os import path
from tensorflow import keras
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import tflearn
img = 'C:\\Users\\84343\\Desktop\\PalmveinBiometrics-master\\dataset\\0001\\0001_m_l_01.jpg'
def processing_image(directory):
    img = cv2.imread(directory)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m,n =  img.shape

    img_new = np.zeros([m, n]) 

    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = [img[i-1, j-1], 
                img[i-1, j], 
                img[i-1, j + 1], 
                img[i, j-1], 
                img[i, j], 
                img[i, j + 1], 
                img[i + 1, j-1], 
                img[i + 1, j], 
                img[i + 1, j + 1]] 
            
            temp = sorted(temp) 
            img_new[i, j]= temp[4] 
    
    img_new = img_new.astype(np.uint8) 
    # cv2.imwrite('new_median_filtered.png', img_new) 
    # cv2.imshow('Filter image',img_new) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_new = cv2.equalizeHist(img_new)

    #stacking images side-by-side
    # cv2.imshow('histogram equalization',img_new) 
    # cv2.waitKey(0)

    image = invert(img_new,0)

    img = image.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    iterations = 0

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    ret, thr = cv2.threshold(skel, 5,255, cv2.THRESH_BINARY)
    thr = cv2.resize(thr,(64,48))
    
    #cv2.imshow(directory,thr) 
    #cv2.waitKey(0)
    thr = thr.reshape(1, 48*64)
    return thr


def extractdatabase(LeftOrRight,MaleOrFemale,scale):
    trainx = np.array([], dtype=np.int64).reshape(0,3072)
    trainy = np.array([], dtype=np.int64).reshape(0,10)
    testx = np.array([], dtype=np.int64).reshape(0,3072)
    testy = np.array([], dtype=np.int64).reshape(0,10)
    for i in range(1,10):
        subject = str(i).zfill(4)
        b=np.zeros(10)
        b[int(subject)-1]=1
        b=b.reshape(1,10)
        if LeftOrRight == 'Left':
            LoR = 'l'
        elif LeftOrRight == 'Right':
            LoR = 'r'
        if MaleOrFemale == 'Male':
            MoF = 'm'
        elif MaleOrFemale == 'Female':
            MoF = 'f'
        dic = 'C:\\Users\\84343\\Desktop\\PalmveinBiometrics-master\\dataset\\' + \
        subject + '\\' + subject + '_' + MoF + '_' + LoR + '_' + '01' + '.jpg'
        if path.exists(dic) == False:
            continue
        a = processing_image(dic)
        c = b
        trainx = np.concatenate((trainx,a))
        for j in range(2, scale + 1):
            dic = dic[:72] + str(j) + '.jpg'
            x = processing_image(dic)
            a = np.concatenate((trainx,x))
            c = np.concatenate((c,b))
        trainx = np.vstack([trainx, a])
        trainy = np.vstack([trainy,c])
        test_b = b

        dic = 'C:\\Users\\84343\\Desktop\\PalmveinBiometrics-master\\dataset\\' + \
        subject + '\\' + subject + '_' + MoF + '_' + LoR + '_' + '0' + str(scale+1) + '.jpg'
        test_a = processing_image(dic)
        trainingSeries = 7 - scale
        for k in reversed(range(9-trainingSeries,9)):
            dic = dic[:72] + str(k) + '.jpg'
            x = processing_image(dic)
            test_a = np.concatenate((test_a,x))
            test_b = np.concatenate((test_b,b))
        
        testx=np.vstack([testx,test_a])
        testy=np.vstack([testy,test_b])
    return trainx,trainy,testx,testy

#X_train,Y_train,X_test,Y_test = extractdatabase('Left','Male',6)

# model = Sequential()
# # convolutional layer
# model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(64,48,0)))
# model.add(MaxPool2D(pool_size=(1,1)))
# # flatten output of conv
# model.add(Flatten())
# # hidden layer
# model.add(Dense(100, activation='relu'))
# # output layer
# model.add(Dense(10, activation='softmax'))

# # compiling the sequential model
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # training the model for 10 epochs
# model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
# print(accuracy_score)

def new_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))
    
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filters=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(input=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    
    images = data.test.images[incorrect]
    
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):

    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = data.test.images[i:j, :]

        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
    
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:

            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
def plot_conv_layer(layer, image):
    feed_dict = {x: [image]}

    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
from sklearn.model_selection import train_test_split

#X_train, X_test,Y_train,Y_test = extractdatabase('Left','Male',6)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LeftOrRight = 'Left'
MaleOrFemale = 'Male'
scale = 6
trainx, trainy, testx,testy = extractdatabase(LeftOrRight,MaleOrFemale,scale)
print(trainy.size)

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)
network = input_data(shape=[None, 48,64,1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule-classifier.tfl.ckpt')
model.fit(trainx, trainy, n_epoch=100, shuffle=True, validation_set=(testx, testy),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='nodule-classifier')

model.save("nodule-classifier.tfl")
print("Network trained and saved as nodule-classifier.tfl!")