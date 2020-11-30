import numpy as np
import tensorflow as tf
import os

from preprocess import get_data_main
from vgg_model import PseudoVGG

def train(model, train_inputs, train_labels, verbose=False):
    BATCH_SZ = model.batch_size
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    indices = np.arange(train_inputs.shape[0]).tolist()
    #random.shuffle(indices)
    loss_list = []
    for i in range(0, train_labels.shape[0], BATCH_SZ):
        images = train_inputs[indices[i:i + BATCH_SZ]]
        #image = image[:, :, :, np.newaxis]
        labels = tf.gather(train_labels, indices[i:i + BATCH_SZ])
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = model.loss_function(labels, logits)
            if i//BATCH_SZ % 4 == 0 and verbose:
                train_acc = loss
                print("Accuracy on training set after {} training steps: {}".format(i, train_acc))

        # The keras Model class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
        loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model.loss_list

def test():
	pass

def main():
	path = '../data/main_dataset/'
	train_data, train_labels = get_data_main(path + 'train/')
	test_data, test_labels = get_data_main(path + 'test/')

	model = PseudoVGG()
	train(model, train_data, train_labels, verbose=True)
	

if __name__ == '__main__':
	main()
