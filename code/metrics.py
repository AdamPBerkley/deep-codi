import tensorflow as tf
import numpy as np



def dice_coef(y_true, y_pred, smooth=1e-10):
    """ dice_coef =  2*TP/(|pred|+|true|)
    the code below works because labels are one-hot enconded."""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, 'float32'))
    return (2. * tf.reduce_sum(y_true_f * y_pred_f)) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def specificity(y_true, y_pred):
    """
    returns: float 
    """
    # m = tf.keras.metrics.FalsePositives(threshold)
    # m.update_state(y_true,y_pred)
    # fp = m.result().numpy()

    # n = tf.keras.metrics.TrueNegatives(threshold)
    # n.update_state(y_true,y_pred)
    # tn = m.result().numpy()
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_negatives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1))), tf.float64)
    possible_negatives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1))), tf.float64)
    
    specificity = true_negatives / (possible_negatives + 1.0e-7)
    
    return specificity
    
def sensitivity(y_true, y_pred):
    """
    returns: float 
    """
    # m = tf.keras.metrics.FalseNegatives(threshold)
    # m.update_state(y_true,y_pred)
    # fn = m.result().numpy()

    # n = tf.keras.metrics.TruePositives(threshold)
    # n.update_state(y_true,y_pred)
    # tp = m.result().numpy()
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_positives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1))), tf.float64)
    possible_positives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1))), tf.float64)
    print(true_positives)
    print(possible_positives)

    sensitivity = true_positives / (possible_positives + 1.0e-7)
    
    return sensitivity

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
    y_true_max = tf.argmax(y_true, axis=-1)
    weighted_true = tf.gather(weights, y_true_max)
    loss = tf.keras.metrics.categorical_crossentropy(y_pred, y_true) * weighted_true
    return loss 

if __name__ == '__main__':
    true = np.array([[1,0],[0,1]])
    pred = np.array([[0.9,0.1],[0,1]])
    print(specifictiy(true, pred))
    print(sensitivity(true,pred))


