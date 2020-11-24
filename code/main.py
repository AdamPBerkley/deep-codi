import numpy as np
import tensorflow as tf
import os

from preprocess import get_data_main
from PIL import Image


def main():
	path = '../data/main_dataset/'
	train_data, train_labels = get_data_main(path + 'train/')
	# test_data, test_labels = get_data_main(path + 'test/')

	jpg = train_data[0]
	png = train_data[1]
	print(np.array_equal(jpg, png))
	print(np.sum(jpg - png))
	
	# from matplotlib import pyplot as plt
	# plt.imshow(train_data[0], interpolation='nearest')
	# plt.show()

	# plt.imshow(train_data[2], interpolation='nearest')
	# plt.show()

if __name__ == '__main__':
	main()
