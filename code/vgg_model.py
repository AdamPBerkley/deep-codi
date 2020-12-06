import numpy as np
import tensorflow as tf
from metrics import dice_coef, sensitivity

#Layers based on VGG structure
#Filters/biases in VGG come from Numpy file
#Pseudo/skeleton code

class PseudoVGG(tf.keras.Model):
    def __init__(self):
        super(PseudoVGG, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.batch_size = 32
        self.epochs = 5
        self.color = 'RGB' #should be 'RGB' or 'L'
        kernel_size = 6
        
        self.conv1_1 = tf.keras.layers.Conv2D(64,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv1_2 = tf.keras.layers.Conv2D(64,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
        
        self.conv2_1 = tf.keras.layers.Conv2D(128,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv2_2 = tf.keras.layers.Conv2D(128,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
        
        self.conv3_1 = tf.keras.layers.Conv2D(256,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv3_2 = tf.keras.layers.Conv2D(256,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv3_3 = tf.keras.layers.Conv2D(256,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
            
        self.conv4_1 = tf.keras.layers.Conv2D(512,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv4_2 = tf.keras.layers.Conv2D(512,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv4_3 = tf.keras.layers.Conv2D(512,kernel_size,activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
            
        self.conv5_1 = tf.keras.layers.Conv2D(512,kernel_size, activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv5_2 = tf.keras.layers.Conv2D(512,kernel_size, activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.conv5_3 = tf.keras.layers.Conv2D(512,kernel_size, activation='relu', padding='SAME',use_bias=True,bias_initializer="zeros")
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
        self.flatten = tf.keras.layers.Flatten()
            
        self.dense6 = tf.keras.layers.Dense(4096, use_bias=True, activation='relu', bias_initializer="zeros")
        self.dense7 = tf.keras.layers.Dense(1000, use_bias=True, activation='relu', bias_initializer="zeros")
        self.dense8 = tf.keras.layers.Dense(2, use_bias=True, activation=None, bias_initializer="zeros")
    
    def call(self,covid_input):
        conv1_1 = self.conv1_1(covid_input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
    
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_2(conv3_2)
        pool3 = self.pool3(conv3_2)   

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.conv4_2(conv4_2)
        pool4 = self.pool4(conv4_2)  

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        pool5 = self.pool5(conv5_3)
        flat = self.flatten(pool5)

        dense6 = self.dense6(flat)
        dense7 = self.dense7(dense6)
        dense8 = self.dense8(dense7)
        
        probs = tf.nn.softmax(dense8)        
        return probs

    def loss_function(self, y_true, y_pred):
        """self.bce is binary crossentropy while self.cce is categorical crossentropy
        used both loss types because I've been switching between them to try and improve
        the model"""
        crossentropy = self.cce(y_true, y_pred)
        return crossentropy #- dice_coef(y_true, y_pred)

