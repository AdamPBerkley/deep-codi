import numpy as np
import tensorflow as tf
import os
import pandas as pd

from balanced_gen import BalancedDataGenerator
from metrics import *


from preprocess import get_data_main
from PIL import Image



def train(model,train_data,train_labels):
 #Create Training Data Generator for augmentation
    CSVLogger = tf.keras.callbacks.CSVLogger('train_logs.csv',separator=",")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.25],
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
        
    seed = 1

    #Feed Training data and training data generator into Balanced Data Generator: augments data such that it is not heavily imbalanced
    balanced_gen = BalancedDataGenerator(train_data, train_labels, train_datagen,batch_size = 32)
 
    train_steps = balanced_gen.steps_per_epoch
    
    #Stop Early if val_accuracy no longer improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        mode='min',
        patience=5
        )
        
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights.{epoch:02d}-{loss:.2f}.hdf5',
        save_weights_only=True,
        save_freq = 'epoch',
        monitor='loss',
        mode='min',
        save_best_only=True)
        
    #Fit Model
    model.fit_generator(
        balanced_gen,
        steps_per_epoch=train_steps,
        epochs=10,
        callbacks=[early_stopping,checkpoint_callback,CSVLogger]
        )   
     
    
def test(model,test_path):
    seed = 1
    CSVLogger = tf.keras.callbacks.CSVLogger('test_logs.csv',separator=",")
    #Create Test Generator
    testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
    testing_generator =testing_datagen.flow_from_directory(
        test_path,
        color_mode='rgb',
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        seed=seed
        ) 
    test_steps = testing_generator.n//testing_generator.batch_size
    
    model.evaluate_generator(testing_generator,
    steps=test_steps, callbacks = [CSVLogger],
    verbose=1)        

    
    
  
    
def main():
    train_path = '../data/main_dataset/train/'
    test_path ='../data/main_dataset/test/'
    print("Loading the data...")
    
    #Load Training Data
    train_data, train_labels = get_data_main(train_path)
    print(len(train_data),len(train_labels))
 
    print("Generating the model...")
    shape = (224, 224, 3)
    vgg16 = tf.keras.applications.VGG16(input_shape=shape, include_top=False, weights='imagenet')
    vgg16.trainable=False
    
    model = tf.keras.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))
    model.compile(optimizer=tf.optimizers.Adam(.0001), loss='binary_crossentropy',run_eagerly=True,metrics=["accuracy",sensitivity,specificity,precision,dice_coef])
    model.summary()
    
    


    print("Training...")
    train(model,train_data,train_labels)    
    model.save("../models/")
    
    print("Testing...")
    test(model,test_path)




if __name__ == '__main__':
    main()
