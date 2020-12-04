import numpy as np
import tensorflow as tf
import os
import pandas as pd

from balanced_gen import BalancedDataGenerator

from preprocess import get_data_main
from PIL import Image

batch_size = 20
threshold = .1

def train(model,train_data,train_labels,val_path):
 #Create Training Data Generator for augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.25],
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
        
    seed = 1

    #Feed Training data and training data generator into Balanced Data Generator: augments data such that it is not heavily imbalanced
    balanced_gen = BalancedDataGenerator(train_data, train_labels, train_datagen, batch_size=32)
 
    train_steps = balanced_gen.steps_per_epoch
    
    #Stop Early if val_accuracy no longer improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5
        )
    #Create Validation Data Generator
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
        

    validation_generator = validation_datagen.flow_from_directory(
        val_path,
        color_mode='rgb',
        target_size=(224,224),
        batch_size=16,
        class_mode='binary',
        seed=seed
        )

    valid_steps = validation_generator.n//validation_generator.batch_size        
    #Fit Model
    model.fit_generator(
        balanced_gen,
        steps_per_epoch=train_steps,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=valid_steps,
        callbacks=[early_stopping]
        )   
    
    #Run on Validation Data

    print("Validating...")
    results = model.evaluate_generator(generator=validation_generator,steps=1)    
    print(results)
    
def test(model,test_path):

    #Create Test Generator
    testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
    testing_generator =testing_datagen.flow_from_directory(
        test_path,
        color_mode='rgb',
        target_size=(224,224),
        batch_size=30,
        class_mode='binary',
        seed=seed
        ) 
    test_steps = testing_generator.n//testing_generator.batch_size
    
    model.evaluate_generator(testing_generator,
    steps=test_steps,
    verbose=1)        
def specificity(y_true,y_pred):
    """
    returns: float 
    """
    # m = tf.keras.metrics.FalsePositives(threshold)
    # m.update_state(y_true,y_pred)
    # fp = m.result().numpy()

    # n = tf.keras.metrics.TrueNegatives(threshold)
    # n.update_state(y_true,y_pred)
    # tn = m.result().numpy()   
    tn = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    
    specificity = tn / (fp + 1e-7)
    
    return specificity
    
    
    
def sensitivity(y_true,y_pred):
    """
    returns: float 
    """
    # m = tf.keras.metrics.FalseNegatives(threshold)
    # m.update_state(y_true,y_pred)
    # fn = m.result().numpy()

    # n = tf.keras.metrics.TruePositives(threshold)
    # n.update_state(y_true,y_pred)
    # tp = m.result().numpy()
    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    fn = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))

    sensitivity = tp / (fn + 1e-7)
    
    return sensitivity
    
def main():
    train_path = '../data/main_dataset/train/'
    test_path ='../data/main_dataset/test/'
    val_path = '../data/main_dataset/validation/'
    print("Loading the data...")
    
    #Load Training Data
    train_data, train_labels = get_data_main(train_path)
    
 
    #print(testing_generator.filenames)
    print("Generating the model...")
    shape = (224, 224, 3)
    vgg16 = tf.keras.applications.VGG16(input_shape=shape, include_top=False, weights='imagenet')
    vgg16.trainable=False
    pool_layer = tf.keras.layers.GlobalAveragePooling2D()
    pred_layer = tf.keras.layers.Dense(1,activation='sigmoid')
    
    model = tf.keras.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))
    model.compile(optimizer=tf.optimizers.Adam(.0001), loss='binary_crossentropy',run_eagerly=True,metrics=["accuracy",sensitivity,specificity])
    model.summary()
    
    


    print("Training...")
    train(model,train_data,train_labels,val_path)    

    
    print("Testing...")
    test(model,test_path)



    model.save("../models/")
if __name__ == '__main__':
    main()
