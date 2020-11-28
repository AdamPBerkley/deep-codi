import numpy as np
import tensorflow as tf
import os

from preprocess import get_data_main
from PIL import Image

batch_size = 20
threshold = .1 #below = not covid, above = covid

def test(model,test_data):
    preds = model.predict(test_data,batch_size = batch_size)
    return preds

def specificty(test_labels,predictions,threshold):
    """
    returns: float that represents # of correctly predicted non-COVID images over the total number of non-COVID images
    """
    correct = 0
    total = 0
    for i in range(len(predictions)):
        if np.argmax(test_labels[i]) == 0: #Non-COVID image
            total += 0
            if predictions[i] < threshold: #Correct prediction of Non-COVID Image
                correct +=1
    return correct/total
def sensitivity(test_labels,predictions,threshold):
    """
    returns: float that represents # of correctly predicted COVID images over the total number of COVID images
    """
    correct = 0
    total = 0
    for i in range(len(predictions)):
        if np.argmax(test_labels[i]) == 1: #COVID image
            total += 0
            if predictions[i] >= threshold: #Correct prediction of COVID Image
                correct +=1
    return correct/total

def main():
    path = '../data/main_dataset/'
    print("Loading the data...")
    train_data, train_labels = get_data_main(path + 'train/')
    test_data, test_labels = get_data_main(path + 'test/')
    
    print("Generating the model...")
    vgg16 = tf.keras.applications.VGG16(include_top=True, weights="imagenet")

    sensitivity_list = []
    specificity_list = []
    
    print("Testing...")
    for start, end in zip(range(0, len(test_data) - batch_size, batch_size), range(batch_size, len(test_data), batch_size)):
        test_batch = np.expand_dims(test_data[start:end],axis=0)
        batch_labels = test_labels[start:end]       
        predictions = test(vgg16,test_batch)
        sensitivity_list.append(sensitivity(batch_labels,predictions,threshold))
        specificity_list.append(specificity(batch_labels,predictions,threshold))
    tot_spec = np.mean(specificity_list)
    tot_sens = np.mean(sensitivity_list)
    print('Avg Specificity: ', tot_spec)
    print('Avg Sensitivity: ', tot_sens)
    
if __name__ == '__main__':
    main()
