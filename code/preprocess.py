import pickle
import numpy as np
import tensorflow as tf
import os
import glob

from PIL import Image
from balanced_gen import BalancedDataGenerator

def normalize_image(image):
    image = image/np.max(image)
    return image


def get_balanced_data(path, imsize=224, batch_size=32, color='L'):
    inputs, labels =  get_data_main(path, imsize=imsize, color=color)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.25],
        preprocessing_function=normalize_image if color == 'L' else tf.keras.applications.vgg16.preprocess_input
        #tf.keras.applications.vgg16.preprocess_input
        #^^ will work if we convert to RBG instead of L in get_data_main
        )
        
    seed = 1

    #Feed Training data and training data generator into Balanced Data Generator: augments data such that it is not heavily imbalanced
    balanced_gen = BalancedDataGenerator(inputs, labels, datagen, batch_size=batch_size)

    return balanced_gen


def get_data_main(path, imsize=224, oversample=1, color='L'):
    """
    Given a file path, returns an array of normalized inputs (images) and an array of 
    one_hot encoded binary labels. 

    :param path: file path for inputs and labels, something 
    like '../data/main_dataset/train/'
    :param imsie:  an integer  representing how many pixels you want in your 
    image for the dataset. uses Pillow for automatic scaling to correct size 
    :return:NumPy array of normalized inputs of labels, where 
    inputs are black and white (one channel) and of size imsize x imsize    """
    covid_pics = glob.glob(path+"covid/*")
    if 'test' in path:
        non_covid_pics = glob.glob(path+"non/**/*")
    else:
        non_covid_pics = glob.glob(path+"non/*")
    num_pics = len(covid_pics)*oversample+len(non_covid_pics)
    if color == 'L':
        data = np.empty((num_pics, imsize, imsize, 1))
    else:
        data = np.empty((num_pics, imsize, imsize, 3))
    labels = np.zeros((num_pics, 2))
    index = 0
    for i in range(oversample):
        for pic in covid_pics:
            image = Image.open(pic).resize((imsize,imsize)).convert(color)
            im_data = np.asarray(image)
            if color == 'L':
                data[index] = np.expand_dims(normalize_image(im_data), -1)
            else:
                data[index] = normalize_image(im_data)
            labels[index,1] = 1
            index += 1
    for pic in non_covid_pics:
        image = Image.open(pic).resize((imsize,imsize)).convert(color)
        im_data = np.asarray(image)
        if color == 'L':
            data[index] = np.expand_dims(normalize_image(im_data), -1)
        else:
            data[index] = normalize_image(im_data)
        labels[index,0] = 1
        index += 1

    return data, labels


if __name__ == '__main__':
    path1 = '../data/kaggle_dataset/train/*'
    path2 = '../data/main_dataset/train/'
    data, labels = get_data_main(path2)
