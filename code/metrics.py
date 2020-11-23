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


def true_dice_for_2D_slice(inputs, y_true, y_pred, smooth=1e-10):
    """ above dice coefficient assumes all values are from inside the brain. this 
    function check and discludes values outside the brain in the accuracy. only works on
    2D slice.
    note: doesn't work while training, but all patches are from inside brain then anyways"""

    one_channel = inputs[:,:,1]
    one_channel_f = one_channel.astype('float32').flatten()
    indices = np.where(one_channel_f == one_channel_f[0])
    mask = np.ones(one_channel_f.shape)
    mask[indices] = False
    mask = mask.astype('int')
    total = np.sum(mask)
    if(len(np.where(mask==1)[0])<=5):
        return None, 0
    y_true_f = y_true[mask].astype('float32').flatten()
    y_pred_f = y_pred[mask].astype('float32').flatten()
    dice = (2. * np.dot(y_true_f, y_pred_f)) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice, total

def true_acc_for_2D_slice(inputs, y_true, y_pred):
    """ above dice coefficient assumes all values are from inside the brain. this 
    function check and discludes values outside the brain in the accuracy. only works on
    2D slice.
    note: doesn't work while training, but all patches are from inside brain then anyways"""

    one_channel = inputs[:,:,1]
    one_channel_f = one_channel.astype('float32').flatten()
    indices = np.where(one_channel_f == one_channel_f[0])
    mask = np.ones(one_channel_f.shape)
    mask[indices] = False
    mask = mask.astype('int')
    total = np.sum(mask)
    if(total<=5):
        return None, 0
    y_true_f = y_true[mask].astype('float32').flatten()
    y_pred_f = y_pred[mask].astype('float32').flatten()
    acc = (np.dot(y_true_f, y_pred_f)) / (y_true.shape[0])
    return acc, total

def accuracy_helper(y_true, y_pred, label, smooth=1e-10):
    assert(y_true.shape == y_pred.shape)
    y_true_f = K.argmax(y_true, axis=-1)
    length = K.int_shape(y_true_f)[0]
    #if length == None or not length.is_integer():
    #    length = 0
    y_pred_f = K.argmax(y_pred, axis=-1)
    mask = K.constant(label, shape=(length,), dtype='int64')
    y_true_bools = K.equal(y_true_f, mask)
    y_pred_bools = K.equal(y_pred_f, mask)
    true_positives = K.sum(K.cast(y_true_bools,'float32')*K.cast(y_pred_bools,'float32'))
    total = K.sum(K.cast(y_true_bools, 'float32'))
    return true_positives/(total + smooth)

def dice_helper(y_true, y_pred, label, smooth=1e-10):
    assert(y_true.shape == y_pred.shape)
    y_true_f = K.argmax(y_true, axis=-1)
    y_pred_f = K.argmax(y_pred, axis=-1)
    shape = y_true_f.shape
    if shape[0] == None:
        shape = (0,)
    mask = K.constant(label, shape=shape, dtype='int64')
    y_true_bools = K.equal(y_true_f, mask)
    y_pred_bools = K.equal(y_pred_f, mask)
    true_positives = K.sum(K.cast(y_true_bools,'float32')*K.cast(y_pred_bools,'float32'))
    total_true = K.sum(K.cast(y_true_bools, 'float32'))
    #total for prediction is TP + FP, so total-TP = FP
    total_pred = K.sum(K.cast(y_pred_bools, 'float32'))
    false_positives = total_pred - true_positives
    #2*TP/(2(TP + FP + FN))
    #total = TP + FN
    return 2.0*true_positives/(total_true + false_positives + smooth)

def L1_acc(y_true, y_pred):
    return accuracy_helper(y_true, y_pred, 1)

def L2_acc(y_true, y_pred):
    return accuracy_helper(y_true, y_pred, 2)

def L4_acc(y_true, y_pred):
    return accuracy_helper(y_true, y_pred, 3)


def L1_dice(y_true, y_pred):
    return dice_helper(y_true, y_pred, 1)

def L2_dice(y_true, y_pred):
    return dice_helper(y_true, y_pred, 2)

def L4_dice(y_true, y_pred):
    return dice_helper(y_true, y_pred, 3)

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