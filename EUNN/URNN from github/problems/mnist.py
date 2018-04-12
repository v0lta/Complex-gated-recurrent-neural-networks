import numpy as np
from .dataset import Dataset
from tensorflow.examples.tutorials.mnist import input_data

# single output = False
# num_in = 784
# num_target = 1

class MnistProblemDataset(Dataset):

    def __init__(self, num_samples, sample_len):
        super(MnistProblemDataset,self).__init__(num_samples, sample_len)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.X_train = np.reshape(mnist.train.images, [-1, 784, 1])
        self.Y_train = np.reshape(mnist.train.labels, [-1, 1])
        self.X_valid = np.reshape(mnist.validation.images, [-1, 784, 1])
        self.Y_valid = mnist.validation.labels
        self.X_test = np.reshape(mnist.test.images, [-1, 784, 1])
        self.Y_test = mnist.test.labels

    def generate(self, num_samples):
        pass 