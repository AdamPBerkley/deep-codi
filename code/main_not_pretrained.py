import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt

from preprocess import get_data_main, get_balanced_data
from vgg_model import PseudoVGG
from metrics import dice_coef, specificity, sensitivity, precision

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def train(model, generator, verbose=False):
    """trains the model for one epoch

    :param model: tf.keras.Model inherited data type
        model being trained 
    :param generator: BalancedDataGenerator
        a datagenerator which runs preprocessing and returns batches accessed
        by integers indexing (i.e. generator[0] returns the first batch of inputs 
        and labels)
    :param verbose: boolean
        whether to output the dice score every batch
    :return: list
        list of losses from every batch of training
    """
    BATCH_SZ = model.batch_size
    train_steps = generator.steps_per_epoch
    loss_list = []
    for i in range(0, train_steps, 1):
        images, labels = generator[i]
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = model.loss_function(labels, logits)
        if i % 4 == 0 and verbose:
            sensitivity_val = sensitivity(labels, logits)
            specificity_val = specificity(labels, logits)
            precision_val = precision(labels, logits)
            train_dice = dice_coef(labels, logits)
            print("Scores on training batch after {} training steps".format(i))
            print("Sensitivity: {}, Specificity: {}".format(sensitivity_val, specificity_val))
            print("Precision: {}, DICE: {}\n".format(precision_val, train_dice))

        loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_list

def test(model, test_inputs, test_labels):
    """
    :param model: tf.keras.Model inherited data type
        model being trained  
    :param test_input: Numpy Array - shape (num_images, imsize, imsize, channels)
        input images to test on
    :param test_labels: Numpy Array - shape (num_images, 2)
        ground truth labels one-hot encoded
    :return: float, float, float, float 
        returns dice score, sensitivity value, specificity value, 
        and precision value all of which are in the range [0,1]
    """
    BATCH_SZ = model.batch_size
    indices = np.arange(test_inputs.shape[0]).tolist()
    all_logits = None
    for i in range(0, test_labels.shape[0], BATCH_SZ):
        images = test_inputs[indices[i:i + BATCH_SZ]]
        logits = model(images)
        if type(all_logits) == type(None):
            all_logits = logits
        else:
            all_logits = np.concatenate([all_logits, logits], axis=0)
    
    """this should break if the dataset size isnt divisible by the batch size because
    the for loop it runs the batches on doesnt get predictions for the remainder"""

    dice = dice_coef(test_labels, all_logits)
    y_true = tf.argmax(test_labels, axis=-1)
    y_pred = tf.argmax(all_logits, axis=-1)
    sensitivity_val = sensitivity(y_true, y_pred)
    specificity_val = specificity(y_true, y_pred)
    precision_val = precision(y_true, y_pred)

    return dice.numpy(), sensitivity_val.numpy(), specificity_val.numpy(), precision_val.numpy()


def main():    
    model = PseudoVGG()

    path = '../data/main_dataset/'
    #train_data, train_labels = get_data_main(path + 'train/', imsize=224, oversample=5)#~7 for even
    train_generator = get_balanced_data(path + 'train/', imsize=224, batch_size=model.batch_size, color=model.color)
    test_data, test_labels = get_data_main(path + 'test/', imsize=224, oversample=1, color=model.color)#30 for even

    num_epochs = model.epochs
    percent = 0
    losses = []
    for epoch in range(num_epochs):
        #losses += train_old(model, train_data, train_labels, True)
        losses = train(model, train_generator, True)
        curr = int(100* epoch/num_epochs)
        if (curr> percent):
            percent = curr
            print("Completion: {0:.0f}%".format(percent))
    
    print(test(model, test_data, test_labels))
    visualize_loss(losses)



if __name__ == '__main__':
    main()
