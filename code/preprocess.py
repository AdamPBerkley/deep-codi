import pickle
import numpy as np
import tensorflow as tf
import os
import glob

from PIL import Image

def normalize_image(image):
	image = image/np.max(image)
	return image


def get_data_main(path, imsize=224, oversample=1):
	"""
	Given a file path, returns an array of normalized inputs (images) and an array of 
	one_hot encoded binary labels. 

	:param path: file path for inputs and labels, something 
	like '../data/main_dataset/train/'
	:param imsie:  an integer  representing how many pixels you want in your 
	image for the dataset. uses Pillow for automatic scaling to correct size 
	:return:NumPy array of normalized inputs of labels, where 
	inputs are black and white (one channel) and of size imsize x imsize	"""
	covid_pics = glob.glob(path+"covid/*")
	if 'test' in path:
		non_covid_pics = glob.glob(path+"non/**/*")
	else:
		non_covid_pics = glob.glob(path+"non/*")
	num_pics = len(covid_pics)*oversample+len(non_covid_pics)
	data = np.empty((num_pics, imsize, imsize, 1))
	labels = np.zeros((num_pics, 2))
	index = 0
	sizes = []
	for i in range(oversample):
		for pic in covid_pics:
			image = Image.open(pic).convert('L').resize((imsize,imsize))
			sizes.append(image.size)
			im_data = np.asarray(image)
			data[index] = np.expand_dims(normalize_image(im_data), -1)
			labels[index,1] = 1
			index += 1
	for pic in non_covid_pics:
		image = Image.open(pic).convert('L').resize((imsize,imsize))
		sizes.append(image.size)
		im_data = np.asarray(image)
		data[index] = np.expand_dims(normalize_image(im_data), -1)
		labels[index,0] = 1
		index += 1

	return data, labels


if __name__ == '__main__':
	path1 = '../data/kaggle_dataset/train/*'
	path2 = '../data/main_dataset/train/'
	data, labels = get_data_main(path2)
