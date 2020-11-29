import numpy as np
import tensorflow as tf
import os

from preprocess import get_data_main
from PIL import Image

batch_size = 20


def train(model,train_data,train_labels,class_weights):
    model.fit(x=train_data,y=train_labels,batch_size = batch_size,class_weight = class_weights)
    
def test(model,test_data):
    preds = model.predict(test_data,batch_size = batch_size)
    return preds

def specificity(test_labels,predictions):
    """
    returns: float 
    """
    neg_y_true = 1 - test_labels
    neg_y_pred = 1 - predictions
    #Total False Positives: Where Prediction = 1 and Label = 0
    fp = tf.reduce_sum(neg_y_true * predictions)
    #Total True Negatives: Where Prediction = 0 and Label = 0
    tn = tf.reduce_sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + 1e-7)
    
    return specificity
    
    
    
def sensitivity(test_labels,predictions):
    """
    returns: float 
    """
    neg_y_pred = 1-predictions
    #Total False Negatives: Where Prediction = 0 and Label = 1
    fn = tf.reduce_sum(test_labels * neg_y_pred)
    #Total True Positives: Where Prediction = 1 and Label = 1
    tp = tf.reduce_sum(test_labels * predictions)
    sensitivity = tp / (tp + fn + 1e-7)
    return sensitivity
def main():
    path = '../data/main_dataset/'
    print("Loading the data...")
    train_data, train_labels = get_data_main(path + 'train/')
    test_data, test_labels = get_data_main(path + 'test/')
    t=0.1
    covid_samples = np.sum(train_labels)
    non_covid_samples = len(train_labels) - covid_samples
    class_weights={
    0: (non_covid_samples / covid_samples) * len(train_labels) * (1-t),
    1: (non_covid_samples / covid_samples) * len(train_labels) * t
    }
    print("Generating the model...")
    shape = (224, 224, 3)
    vgg16 = tf.keras.applications.VGG16(input_shape=shape, include_top=False, weights='imagenet')
    vgg16.trainable=False
    pool_layer = tf.keras.layers.GlobalAveragePooling2D()
    pred_layer = tf.keras.layers.Dense(2,activation='sigmoid')
    
    model = tf.keras.Sequential([vgg16,pool_layer,pred_layer])
    model.compile(optimizer=tf.optimizers.Adam(.0001), loss='binary_crossentropy',metrics=[sensitivity,specificity])
    model.summary()
    
    sensitivity_list = []
    specificity_list = []
    
    
    
    print("Training...")
    for start, end in zip(range(0, len(train_data) - batch_size, batch_size), range(batch_size, len(train_data), batch_size)):
        train_batch = train_data[start:end]
        batch_labels = train_labels[start:end] 
        train(model,train_batch,batch_labels,class_weights)
        
    # print("Testing...")
    # for start, end in zip(range(0, len(test_data) - batch_size, batch_size), range(batch_size, len(test_data), batch_size)):
        # test_batch = test_data[start:end]
        # batch_labels = test_labels[start:end]       
        # predictions = test(model,test_batch)
        # sensitivity_list.append(sensitivity(batch_labels,predictions,threshold))
        # specificity_list.append(specificity(batch_labels,predictions,threshold))   
    # tot_spec = np.mean(specificity_list) 
    # tot_sens = np.mean(sensitivity_list)
    # print('Avg Specificity: ', tot_spec)
    # print('Avg Sensitivity: ', tot_sens)
    
if __name__ == '__main__':
    main()
