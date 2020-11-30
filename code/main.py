import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt

from preprocess import get_data_main
from vgg_model import PseudoVGG
from metrics import dice_coef, specifictiy, sensitivity 

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

def train(model, train_inputs, train_labels, verbose=False):
    BATCH_SZ = model.batch_size
    indices = np.arange(train_inputs.shape[0]).tolist()
    random.shuffle(indices)
    loss_list = []
    for i in range(0, train_labels.shape[0], BATCH_SZ):
        images = train_inputs[indices[i:i + BATCH_SZ]]
        labels = tf.gather(train_labels, indices[i:i + BATCH_SZ])
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = model.loss_function(labels, logits)
            if i//BATCH_SZ % 4 == 0 and verbose:
                train_dice = dice_coef(labels, logits)
                print("DICE score on training batch after {} training steps: {}".format(i, train_dice))

        loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_list

def test(model, test_inputs, test_labels):
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
    sensitivity_val = sensitivity(test_labels, all_logits)
    specifictiy_val = specifictiy(test_labels, all_logits)

    return dice, sensitivity_val, specifictiy_val


def main():
    path = '../data/main_dataset/'
    train_data, train_labels = get_data_main(path + 'train/')
    test_data, test_labels = get_data_main(path + 'test/')
    
    model = PseudoVGG()
    num_epochs = 1
    percent = 0
    losses = []
    for epoch in range(num_epochs):
        losses += train(model, train_data, train_labels, True)
        curr = int(100* epoch/num_epochs)
        if (curr> percent):
            percent = curr
            print("Completion: {0:.0f}%".format(percent))
    visualize_loss(losses)
    print(test(model, test_data, test_labels))



if __name__ == '__main__':
    main()
