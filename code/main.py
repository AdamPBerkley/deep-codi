import numpy as np
import tensorflow as tf
import os

from preprocess import get_data_main
from PIL import Image


def main():
	path = '../data/main_dataset/'
	train_data, train_labels = get_data_main(path + 'train/')
	test_data, test_labels = get_data_main(path + 'test/')
	

if __name__ == '__main__':
	main()
