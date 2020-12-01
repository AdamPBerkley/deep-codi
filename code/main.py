import numpy as np
import tensorflow as tf
import os

from balanced_gen import BalancedDataGenerator

from preprocess import get_data_main
from PIL import Image

batch_size = 20
threshold = .5

def train(model,train_data,train_labels,class_weights):
    model.fit(x=train_data,y=train_labels,batch_size = batch_size,class_weight=class_weights)
    
def test(model,test_data):
    preds = model.predict(test_data,batch_size = batch_size)
    return preds

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
    steps_per_epoch = balanced_gen.steps_per_epoch
    
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
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5
        )
        
    print("Training...")
    
    #Fit Model
    model.fit_generator(
        balanced_gen,
        steps_per_epoch=50,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=25,
        callbacks=[early_stopping]
        )
    
    

    #for start, end in zip(range(0, len(train_data) - batch_size, batch_size), range(batch_size, len(train_data), batch_size)):
     #   train_batch = train_data[start:end]
      #  batch_labels = train_labels[start:end] 
       # train(model,train_batch,batch_labels,class_weights)
        
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
