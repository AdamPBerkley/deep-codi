from keras import backend as K
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import pdb
import matplotlib.pyplot as plt


def dice_coef(y_true, y_pred, smooth=1e-10):
    """ dice_coef =  2*TP/(|pred|+|true|)
    the code below works because labels are one-hot enconded."""
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    return (2. * K.sum(y_true_f * y_pred_f)) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def specifictiy(y_true, y_pred, smooth=1e-10):
    """TN/(TN+FP)"""
    true_negatives = tf.keras.metrics.TrueNegatives(y_true, y_pred)
    false_positives = tf.keras.metrics.FalsePositives(y_true, y_pred)
    return true_negatives/(true_negatives+false_positives+smooth)

def sensitivity(y_true, y_pred, smooth=1e-10):
    """ TP/ (TP +FN)"""
    true_positives = tf.keras.metrics.TruePositives(y_true, y_pred)
    false_negatives = tf.keras.metrics.FalseNegatives(y_true, y_pred)
    return true_positives/(true_positives+false_negatives+smooth)

def combine_history(history1, history2):
    history = history1
    for key in history1.history:
        history.history[key] = history1.history[key] + history2.history[key]
    return history

def w_categorical_crossentropy(y_true, y_pred, weights):
    """https://www.programcreek.com/python/example/93764/keras.backend.categorical_crossentropy
    Keras-style categorical crossentropy loss function, with weighting for each class.
    Parameters
    ----------
    y_true : Tensor
        Truth labels.
    y_pred : Tensor
        Predicted values.
    weights: Tensor
        Multiplicative factor for loss per class.
    Returns
    -------
    loss : Tensor
        Weighted crossentropy loss between labels and predictions.
    """
    y_true_max = K.argmax(y_true, axis=-1)
    weighted_true = K.gather(weights, y_true_max)
    loss = K.categorical_crossentropy(y_pred, y_true) * weighted_true
    return loss 