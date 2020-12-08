import pickle
import numpy as np
import tensorflow as tf
import os
import glob

from PIL import Image

def normalize_image(image):
    image = image/np.max(image)
    return image


def get_data_main(path, imsize=224):
    """
    Given a file path, returns an array of normalized inputs (images) and an array of 
    one_hot encoded binary labels. 

    :param path: file path for inputs and labels, something 
    like '../data/main_dataset/train/'
    :param imsie:  an integer  representing how many pixels you want in your 
    image for the dataset. uses Pillow for automatic scaling to correct size 
    :return:NumPy array of normalized inputs of labels, where 
    inputs are black and white (one channel) and of size imsize x imsize    """
    covid_pics = glob.glob(path+"1_covid/*")
    if 'test' in path:
        non_pics = glob.glob(path+"0_non/**/*")
    else:
        non_pics = glob.glob(path+"0_non/*")
    num_pics = len(covid_pics)+len(non_pics)
    data = np.empty((num_pics, imsize, imsize, 3))
    labels = np.zeros((num_pics,2))
    index = 0
    sizes = []
    for pic in covid_pics:
        image = Image.open(pic).resize((imsize,imsize)).convert('RGB')
        sizes.append(image.size)
        im_data = np.asarray(image)
        data[index] = normalize_image(im_data)
        labels[index][1] = 1
        index += 1
    for pic in non_pics:
        image = Image.open(pic).resize((imsize,imsize)).convert('RGB')
        sizes.append(image.size)
        im_data = np.asarray(image)
        data[index] = normalize_image(im_data)
        labels[index][0] = 1
        index += 1

    return data, labels


if __name__ == '__main__':
    path1 = '../data/kaggle_dataset/train/*'
    path2 = '../data/main_dataset/train/'
    data, labels = get_data_main(path2)
